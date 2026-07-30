//! Configuration file support for model-rs
//!
//! This module provides functionality to load configuration from TOML and YAML files,
//! in addition to environment variables.

use crate::error::{ModelError, Result};
use crate::validation::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use directories as dirs;
use tokio::fs as tokio_fs;
use tracing::{info, warn, debug};

/// Configuration sources in priority order (highest priority first)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    Environment,
    ConfigFile,
    Default,
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Server configuration
    pub server: ServerConfig,
    /// Generation configuration
    pub generation: GenerationConfig,
    /// Device configuration
    pub device: DeviceConfig,
    /// Download configuration
    pub download: DownloadConfig,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model path
    pub model_path: Option<PathBuf>,
    /// Output directory for downloads
    pub output_dir: Option<PathBuf>,
    /// Auto-pinned models
    pub pinned_models: Vec<String>,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server port
    pub port: u16,
    /// Host address
    pub host: String,
    /// Enable CORS
    pub enable_cors: bool,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationConfig {
    /// Default temperature
    pub temperature: f32,
    /// Default top-p
    pub top_p: f32,
    /// Default top-k
    pub top_k: Option<u32>,
    /// Default repeat penalty
    pub repeat_penalty: f32,
    /// Default max tokens
    pub max_tokens: u32,
    /// Warmup tokens
    pub warmup_tokens: Option<u32>,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Default device
    pub device: String,
    /// Device index
    pub device_index: u32,
    /// Enable Metal acceleration (macOS)
    pub enable_metal: bool,
    /// Enable CUDA acceleration
    pub enable_cuda: bool,
}

/// Download configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadConfig {
    /// Default mirror URL
    pub mirror_url: String,
    /// Download timeout in seconds
    pub timeout_seconds: u64,
    /// Enable progress bar
    pub show_progress: bool,
    /// Concurrent download chunks
    pub max_concurrent_downloads: usize,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Maximum cached models
    pub max_models: usize,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Cache expiration in hours
    pub expiration_hours: u64,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable colored output
    pub colored: bool,
    /// Log to file
    pub log_file: Option<PathBuf>,
    /// Enable JSON logging
    pub json: bool,
}

/// Default configuration
impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                model_path: None,
                output_dir: None,
                pinned_models: Vec::new(),
            },
            server: ServerConfig {
                port: 8080,
                host: "127.0.0.1".to_string(),
                enable_cors: true,
                timeout_seconds: 300,
            },
            generation: GenerationConfig {
                temperature: 0.7,
                top_p: 0.9,
                top_k: Some(50),
                repeat_penalty: 1.1,
                max_tokens: 512,
                warmup_tokens: Some(10),
            },
            device: DeviceConfig {
                device: "auto".to_string(),
                device_index: 0,
                enable_metal: cfg!(target_os = "macos"),
                enable_cuda: cfg!(target_os = "linux"),
            },
            download: DownloadConfig {
                mirror_url: "https://huggingface.co".to_string(),
                timeout_seconds: 600,
                show_progress: true,
                max_concurrent_downloads: 3,
            },
            cache: CacheConfig {
                enabled: true,
                max_models: 5,
                cache_dir: None,
                expiration_hours: 24,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                colored: true,
                log_file: None,
                json: false,
            },
        }
    }
}

/// Configuration manager
#[derive(Clone)]
pub struct ConfigManager {
    config_dir: PathBuf,
    config_file: PathBuf,
    app_config: AppConfig,
    config_sources: HashMap<String, ConfigSource>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Result<Self> {
        let config_dir = Self::get_config_dir()?;
        let config_file = config_dir.join("config.toml"); // Default to TOML
        
        let app_config = Self::load_configuration(&config_file)?;
        
        Ok(Self {
            config_dir,
            config_file,
            app_config,
            config_sources: HashMap::new(),
        })
    }

    /// Get configuration directory
    fn get_config_dir() -> Result<PathBuf> {
        let home_dir = std::env::home_dir()
            .ok_or_else(|| ModelError::invalid_config(
                "Cannot determine home directory. Set HOME environment variable",
            ))?;
        
        let config_dir = home_dir.join(".config").join("model-rs");
        
        // Create directory if it doesn't exist
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .map_err(|e| ModelError::validation_error(
                    config_dir.to_string_lossy().as_ref(),
                    &format!("Failed to create config directory: {}", e),
                    "Check file permissions and disk space"
                ))?;
        }
        
        Ok(config_dir)
    }

    /// Load configuration from file
    fn load_configuration(config_file: &Path) -> Result<AppConfig> {
        if !config_file.exists() {
            info!("Configuration file not found at {}, using defaults", config_file.display());
            return Ok(AppConfig::default());
        }

        let content = fs::read_to_string(config_file)
            .map_err(|e| ModelError::validation_error(
                config_file.to_string_lossy().as_ref(),
                &format!("Failed to read configuration file: {}", e),
                "Check file permissions"
            ))?;

        let config = if config_file.extension().and_then(|s| s.to_str()) == Some("yaml") 
            || config_file.extension().and_then(|s| s.to_str()) == Some("yml") {
            // Load YAML
            serde_yaml::from_str(&content)
                .map_err(|e| ModelError::validation_error(
                    config_file.to_string_lossy().as_ref(),
                    &format!("Invalid YAML configuration: {}", e),
                    "Check the YAML format"
                ))?
        } else {
            // Load TOML (default)
            toml::from_str(&content)
                .map_err(|e| ModelError::validation_error(
                    config_file.to_string_lossy().as_ref(),
                    &format!("Invalid TOML configuration: {}", e),
                    "Check the TOML format"
                ))?
        };

        Ok(config)
    }

    /// Save configuration to file
    pub async fn save_configuration(&mut self, config: &AppConfig) -> Result<()> {
        let content = toml::to_string_pretty(config)
            .map_err(|e| ModelError::validation_error(
                self.config_file.to_string_lossy().as_ref(),
                &format!("Failed to serialize configuration: {}", e),
                "Check configuration values"
            ))?;

        tokio_fs::write(&self.config_file, content).await
            .map_err(|e| ModelError::validation_error(
                self.config_file.to_string_lossy().as_ref(),
                &format!("Failed to write configuration file: {}", e),
                "Check file permissions and disk space"
            ))?;

        info!("Configuration saved to {}", self.config_file.display());
        Ok(())
    }

    /// Get merged configuration (file + environment + defaults)
    pub fn get_merged_config(&mut self) -> Result<AppConfig> {
        let mut merged = AppConfig::default();
        
        // Start with defaults
        merged = AppConfig::default();
        
        // Override with file configuration
        if self.config_file.exists() {
            merged = self.merge_configs(merged, self.app_config.clone());
            self.config_sources.insert("model_path".to_string(), ConfigSource::ConfigFile);
        }
        
        // Override with environment variables
        merged = self.apply_environment_overrides(merged);
        
        // Validate configuration
        self.validate_configuration(&merged)?;
        
        Ok(merged)
    }

    /// Merge two configurations
    fn merge_configs(&self, base: AppConfig, override_config: AppConfig) -> AppConfig {
        AppConfig {
            model: ModelConfig {
                model_path: override_config.model.model_path.or(base.model.model_path),
                output_dir: override_config.model.output_dir.or(base.model.output_dir),
                pinned_models: if override_config.model.pinned_models.is_empty() {
                    base.model.pinned_models
                } else {
                    override_config.model.pinned_models
                },
            },
            server: ServerConfig {
                port: override_config.server.port,
                host: override_config.server.host,
                enable_cors: override_config.server.enable_cors,
                timeout_seconds: override_config.server.timeout_seconds,
            },
            generation: GenerationConfig {
                temperature: override_config.generation.temperature,
                top_p: override_config.generation.top_p,
                top_k: override_config.generation.top_k.or(base.generation.top_k),
                repeat_penalty: override_config.generation.repeat_penalty,
                max_tokens: override_config.generation.max_tokens,
                warmup_tokens: override_config.generation.warmup_tokens.or(base.generation.warmup_tokens),
            },
            device: DeviceConfig {
                device: override_config.device.device,
                device_index: override_config.device.device_index,
                enable_metal: override_config.device.enable_metal,
                enable_cuda: override_config.device.enable_cuda,
            },
            download: DownloadConfig {
                mirror_url: override_config.download.mirror_url,
                timeout_seconds: override_config.download.timeout_seconds,
                show_progress: override_config.download.show_progress,
                max_concurrent_downloads: override_config.download.max_concurrent_downloads,
            },
            cache: CacheConfig {
                enabled: override_config.cache.enabled,
                max_models: override_config.cache.max_models,
                cache_dir: override_config.cache.cache_dir.or(base.cache.cache_dir),
                expiration_hours: override_config.cache.expiration_hours,
            },
            logging: LoggingConfig {
                level: override_config.logging.level,
                colored: override_config.logging.colored,
                log_file: override_config.logging.log_file.or(base.logging.log_file),
                json: override_config.logging.json,
            },
        }
    }

    /// Apply environment variable overrides
    fn apply_environment_overrides(&mut self, mut config: AppConfig) -> AppConfig {
        // Model configuration
        if let Some(model_path) = env::var("MODEL_RS_MODEL_PATH").ok() {
            config.model.model_path = Some(PathBuf::from(model_path));
            self.config_sources.insert("model_path".to_string(), ConfigSource::Environment);
        }
        
        if let Some(output_dir) = env::var("MODEL_RS_OUTPUT_DIR").ok() {
            config.model.output_dir = Some(PathBuf::from(output_dir));
            self.config_sources.insert("output_dir".to_string(), ConfigSource::Environment);
        }
        
        // Server configuration
        if let Some(port) = env::var("MODEL_RS_PORT").ok().and_then(|s| s.parse().ok()) {
            config.server.port = port;
            self.config_sources.insert("port".to_string(), ConfigSource::Environment);
        }
        
        // Generation configuration
        if let Some(temperature) = env::var("MODEL_RS_TEMPERATURE").ok().and_then(|s| s.parse().ok()) {
            config.generation.temperature = temperature;
            self.config_sources.insert("temperature".to_string(), ConfigSource::Environment);
        }
        
        if let Some(top_p) = env::var("MODEL_RS_TOP_P").ok().and_then(|s| s.parse().ok()) {
            config.generation.top_p = top_p;
            self.config_sources.insert("top_p".to_string(), ConfigSource::Environment);
        }
        
        if let Some(top_k) = env::var("MODEL_RS_TOP_K").ok().and_then(|s| s.parse().ok()) {
            config.generation.top_k = Some(top_k);
            self.config_sources.insert("top_k".to_string(), ConfigSource::Environment);
        }
        
        if let Some(repeat_penalty) = env::var("MODEL_RS_REPEAT_PENALTY").ok().and_then(|s| s.parse().ok()) {
            config.generation.repeat_penalty = repeat_penalty;
            self.config_sources.insert("repeat_penalty".to_string(), ConfigSource::Environment);
        }
        
        if let Some(max_tokens) = env::var("MODEL_RS_MAX_TOKENS").ok().and_then(|s| s.parse().ok()) {
            config.generation.max_tokens = max_tokens;
            self.config_sources.insert("max_tokens".to_string(), ConfigSource::Environment);
        }
        
        // Device configuration
        if let Some(device) = env::var("MODEL_RS_DEVICE").ok() {
            config.device.device = device;
            self.config_sources.insert("device".to_string(), ConfigSource::Environment);
        }
        
        if let Some(device_index) = env::var("MODEL_RS_DEVICE_INDEX").ok().and_then(|s| s.parse().ok()) {
            config.device.device_index = device_index;
            self.config_sources.insert("device_index".to_string(), ConfigSource::Environment);
        }
        
        // Download configuration
        if let Some(mirror) = env::var("MODEL_RS_MIRROR").ok() {
            config.download.mirror_url = mirror;
            self.config_sources.insert("mirror_url".to_string(), ConfigSource::Environment);
        }
        
        config
    }

    /// Validate configuration
    pub fn validate_configuration(&self, config: &AppConfig) -> Result<()> {
        // Validate server port
        if config.server.port == 0 || config.server.port > 65535 {
            return Err(ModelError::validation_error(
                "port",
                "Invalid server port",
                "Use a port between 1 and 65535"
            ));
        }
        
        // Validate generation parameters
        if !(0.0..=2.0).contains(&config.generation.temperature) {
            return Err(ModelError::validation_error(
                "temperature",
                "Temperature must be between 0.0 and 2.0",
                "Use a value between 0.0 and 2.0"
            ));
        }
        
        if !(0.0..=1.0).contains(&config.generation.top_p) {
            return Err(ModelError::validation_error(
                "top_p",
                "Top-p must be between 0.0 and 1.0",
                "Use a value between 0.0 and 1.0"
            ));
        }
        
        if let Some(top_k) = config.generation.top_k {
            if top_k == 0 || top_k > 1000 {
                return Err(ModelError::validation_error(
                    "top_k",
                    "Top-k must be between 1 and 1000",
                    "Use a value between 1 and 1000"
                ));
            }
        }
        
        // Validate device configuration
        let valid_devices = ["auto", "cpu", "metal", "cuda", "mlx"];
        if !valid_devices.contains(&config.device.device.as_str()) {
            return Err(ModelError::validation_error(
                "device",
                &format!("Invalid device: {}", config.device.device),
                &format!("Use one of: {}", valid_devices.join(", "))
            ));
        }
        
        Ok(())
    }

    /// Get configuration source information
    pub fn get_config_sources(&self) -> &HashMap<String, ConfigSource> {
        &self.config_sources
    }

    /// Create a configuration file with current settings
    pub async fn create_config_file(&mut self) -> Result<PathBuf> {
        let config_file = self.config_dir.join("config.toml");
        
        if config_file.exists() {
            return Err(ModelError::config_file_error(
                "Configuration file already exists. Use 'model-rs config edit' to modify existing file",
            ));
        }
        
        let config = AppConfig::default();
        self.save_configuration(&config).await?;
        
        Ok(config_file)
    }

    /// Edit configuration file with default editor
    pub async fn edit_config_file(&mut self) -> Result<()> {
        let config_file = &self.config_file;
        
        if !config_file.exists() {
            return Err(ModelError::config_file_error(
                "Configuration file does not exist. Use 'model-rs config init' to create a new configuration file",
            ));
        }
        
        let editor = env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());
        
        let output = std::process::Command::new(editor)
            .arg(config_file)
            .output()
            .map_err(|e| ModelError::config_file_error(
                format!("Failed to open editor: {}. Check if the editor is installed and accessible", e),
            ))?;
        
        if !output.status.success() {
            return Err(ModelError::config_file_error(
                "Editor exited with error. Check configuration file for syntax errors",
            ));
        }
        
        // Reload configuration after editing
        self.app_config = Self::load_configuration(config_file)?;
        info!("Configuration reloaded from {}", config_file.display());
        
        Ok(())
    }
}

/// Global configuration manager instance
pub static CONFIG_MANAGER: once_cell::sync::Lazy<std::sync::Mutex<Option<ConfigManager>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(None));

/// Initialize global configuration
pub fn init_config() -> Result<()> {
    let manager = ConfigManager::new()?;
    CONFIG_MANAGER.lock().unwrap().replace(manager);
    Ok(())
}

/// Get current merged configuration
pub fn get_config() -> Result<AppConfig> {
    let guard = CONFIG_MANAGER.lock().unwrap();
    let manager = guard
        .as_ref()
        .ok_or_else(|| ModelError::invalid_config(
            "Configuration not initialized. Call init_config() first",
        ))?;
    
    manager.clone().get_merged_config()
}

/// Convenience functions to get specific configuration values
pub fn get_model_path() -> Option<PathBuf> {
    get_config().ok().and_then(|c| c.model.model_path)
}

pub fn get_output_dir() -> Option<PathBuf> {
    get_config().ok().and_then(|c| c.model.output_dir)
}

pub fn get_port() -> u16 {
    get_config().ok().map(|c| c.server.port).unwrap_or(8080)
}

pub fn get_temperature() -> f32 {
    get_config().ok().map(|c| c.generation.temperature).unwrap_or(0.7)
}

pub fn get_top_p() -> f32 {
    get_config().ok().map(|c| c.generation.top_p).unwrap_or(0.9)
}

pub fn get_top_k() -> Option<u32> {
    get_config().ok().and_then(|c| c.generation.top_k)
}

pub fn get_repeat_penalty() -> f32 {
    get_config().ok().map(|c| c.generation.repeat_penalty).unwrap_or(1.1)
}

pub fn get_max_tokens() -> u32 {
    get_config().ok().map(|c| c.generation.max_tokens).unwrap_or(512)
}

pub fn get_device() -> String {
    get_config().ok().map(|c| c.device.device).unwrap_or("auto".to_string())
}

pub fn get_device_index() -> u32 {
    get_config().ok().map(|c| c.device.device_index).unwrap_or(0)
}

pub fn get_mirror() -> String {
    get_config().ok().map(|c| c.download.mirror_url).unwrap_or("https://huggingface.co".to_string())
}

pub fn get_models_dir() -> PathBuf {
    let home_dir = std::env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home_dir.join(".cache").join("model-rs").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_manager_creation() {
        let result = ConfigManager::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_configuration() {
        let config = AppConfig::default();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.generation.temperature, 0.7);
        assert_eq!(config.device.device, "auto");
    }

    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();
        
        // Valid configuration
        assert!(ConfigManager::new().and_then(|m| m.validate_configuration(&config)).is_ok());
        
        // Invalid temperature
        config.generation.temperature = 3.0;
        assert!(ConfigManager::new().and_then(|m| m.validate_configuration(&config)).is_err());
    }

    #[test]
    fn test_merge_configs() {
        let base = AppConfig::default();
        let override_config = AppConfig {
            generation: GenerationConfig {
                temperature: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let manager = ConfigManager::new().unwrap();
        let merged = manager.merge_configs(base, override_config);
        
        assert_eq!(merged.generation.temperature, 1.0);
        assert_eq!(merged.server.port, 8080); // Should keep base value
    }
}