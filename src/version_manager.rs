//! Model version management and pinned model support for model-rs
//!
//! This module provides functionality to manage different versions of models,
//! pin specific versions, and handle version conflicts.

use crate::error::{ModelError, Result};
use crate::validation::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::fs as tokio_fs;
use tokio::io::AsyncReadExt;
use chrono::{DateTime, Utc};
use sha2::Digest;
use tracing::{info, warn, debug};

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub model_id: String,
    pub version: String,
    pub resolved_name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub download_date: DateTime<Utc>,
    pub checksum: Option<String>,
    pub is_pinned: bool,
    pub tags: Vec<String>,
}

/// Model version manager
pub struct ModelVersionManager {
    models_dir: PathBuf,
    versions_dir: PathBuf,
    index_file: PathBuf,
}

impl ModelVersionManager {
    /// Create a new model version manager
    pub fn new(models_dir: &Path) -> Result<Self> {
        validate_path(models_dir, "models directory")?;
        
        let versions_dir = models_dir.join("versions");
        let index_file = versions_dir.join("version_index.json");
        
        // Create versions directory if it doesn't exist
        if !versions_dir.exists() {
            fs::create_dir_all(&versions_dir)
                .map_err(|e| ModelError::validation_error(
                    versions_dir.to_string_lossy().as_ref(),
                    &format!("Failed to create versions directory: {}", e),
                    "Check file permissions and disk space"
                ))?;
        }
        
        Ok(Self {
            models_dir: models_dir.to_path_buf(),
            versions_dir,
            index_file,
        })
    }

    /// Load version index
    pub async fn load_version_index(&self) -> Result<Vec<ModelVersion>> {
        if !self.index_file.exists() {
            return Ok(Vec::new());
        }

        let content = tokio_fs::read_to_string(&self.index_file).await
            .map_err(|e| ModelError::validation_error(
                self.index_file.to_string_lossy().as_ref(),
                &format!("Failed to read version index: {}", e),
                "Check file permissions"
            ))?;

        let versions: Vec<ModelVersion> = serde_json::from_str(&content)
            .map_err(|e| ModelError::validation_error(
                self.index_file.to_string_lossy().as_ref(),
                &format!("Invalid version index format: {}", e),
                "Check the JSON format of the version index"
            ))?;

        Ok(versions)
    }

    /// Save version index
    pub async fn save_version_index(&self, versions: &[ModelVersion]) -> Result<()> {
        let content = serde_json::to_string_pretty(versions)
            .map_err(|e| ModelError::validation_error(
                self.index_file.to_string_lossy().as_ref(),
                &format!("Failed to serialize version index: {}", e),
                "Check the model version data"
            ))?;

        tokio_fs::write(&self.index_file, content).await
            .map_err(|e| ModelError::validation_error(
                self.index_file.to_string_lossy().as_ref(),
                &format!("Failed to write version index: {}", e),
                "Check file permissions and disk space"
            ))?;

        info!("Version index saved with {} versions", versions.len());
        Ok(())
    }

    /// Register a new model version
    pub async fn register_version(&self, model_id: &str, version: &str, resolved_name: &str, path: &Path) -> Result<ModelVersion> {
        validate_model_name(model_id)?;
        
        let mut versions = self.load_version_index().await?;
        
        // Check if this version already exists
        if versions.iter().any(|v| v.model_id == model_id && v.version == version) {
            return Err(ModelError::validation_error(
                model_id,
                &format!("Version {} already exists for model {}", version, model_id),
                "Use a different version number or delete the existing version"
            ));
        }
        
        // Get model size
        let size = tokio_fs::metadata(path).await
            .map_err(|e| ModelError::validation_error(
                path.to_string_lossy().as_ref(),
                &format!("Failed to get model size: {}", e),
                "Check if the model directory exists"
            ))?
            .len();
        
        // Calculate checksum
        let checksum = self.calculate_model_checksum(path).await?;
        
        let new_version = ModelVersion {
            model_id: model_id.to_string(),
            version: version.to_string(),
            resolved_name: resolved_name.to_string(),
            path: path.to_path_buf(),
            size_bytes: size,
            download_date: Utc::now(),
            checksum,
            is_pinned: false,
            tags: Vec::new(),
        };
        
        versions.push(new_version.clone());
        self.save_version_index(&versions).await?;
        
        info!("Registered version {} for model {}", version, model_id);
        Ok(new_version)
    }

    /// Get all versions for a model
    pub async fn get_model_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>> {
        let versions = self.load_version_index().await?;
        Ok(versions.into_iter().filter(|v| v.model_id == model_id).collect())
    }

    /// Get the pinned version for a model
    pub async fn get_pinned_version(&self, model_id: &str) -> Option<ModelVersion> {
        let versions = self.load_version_index().await.ok()?;
        versions.into_iter()
            .filter(|v| v.model_id == model_id && v.is_pinned)
            .next()
    }

    /// Pin a specific version
    pub async fn pin_version(&self, model_id: &str, version: &str) -> Result<()> {
        validate_model_name(model_id)?;
        
        let mut versions = self.load_version_index().await?;
        
        // Find and update the version
        for version_info in &mut versions {
            if version_info.model_id == model_id && version_info.version == version {
                version_info.is_pinned = true;
                self.save_version_index(&versions).await?;
                info!("Pinned version {} for model {}", version, model_id);
                return Ok(());
            }
        }
        
        Err(ModelError::validation_error(
            model_id,
            &format!("Version {} not found for model {}", version, model_id),
            "Use 'model-rs versions <model>' to see available versions"
        ))
    }

    /// Unpin a version
    pub async fn unpin_version(&self, model_id: &str, version: &str) -> Result<()> {
        validate_model_name(model_id)?;
        
        let mut versions = self.load_version_index().await?;
        
        // Find and update the version
        for version_info in &mut versions {
            if version_info.model_id == model_id && version_info.version == version {
                version_info.is_pinned = false;
                self.save_version_index(&versions).await?;
                info!("Unpinned version {} for model {}", version, model_id);
                return Ok(());
            }
        }
        
        Err(ModelError::validation_error(
            model_id,
            &format!("Version {} not found for model {}", version, model_id),
            "Use 'model-rs versions <model>' to see available versions"
        ))
    }

    /// List all registered models and their versions
    pub async fn list_models(&self) -> Result<HashMap<String, Vec<ModelVersion>>> {
        let versions = self.load_version_index().await?;
        let mut models: HashMap<String, Vec<ModelVersion>> = HashMap::new();
        
        for version in versions {
            models.entry(version.model_id.clone()).or_insert_with(Vec::new).push(version);
        }
        
        Ok(models)
    }

    /// Get the latest version for a model
    pub async fn get_latest_version(&self, model_id: &str) -> Option<ModelVersion> {
        let versions = self.load_version_index().await.ok()?;
        let model_versions: Vec<ModelVersion> = versions.into_iter()
            .filter(|v| v.model_id == model_id)
            .collect();
        
        // Sort by download date (newest first)
        model_versions.into_iter()
            .max_by_key(|v| v.download_date)
    }

    /// Clean up old versions
    pub async fn cleanup_old_versions(&self, model_id: Option<&str>, keep_latest: usize, keep_pinned: bool) -> Result<Vec<ModelVersion>> {
        let mut versions = self.load_version_index().await?;
        let mut removed_versions = Vec::new();
        
        let target_versions = if let Some(model_id) = model_id {
            versions.iter().filter(|v| v.model_id == model_id).cloned().collect::<Vec<_>>()
        } else {
            versions.clone()
        };
        
        // Keep pinned versions if specified
        let mut keep = if keep_pinned {
            target_versions.iter().filter(|v| v.is_pinned).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        
        // Add latest versions (excluding pinned ones)
        let unpinned_versions: Vec<&ModelVersion> = target_versions.iter()
            .filter(|v| !v.is_pinned)
            .collect();
        
        if unpinned_versions.len() > keep_latest {
            // Sort by date and keep the latest
            let mut sorted = unpinned_versions.clone();
            sorted.sort_by_key(|v| v.download_date);
            keep.extend(sorted[keep_latest..].to_vec());
        }
        
        // Remove versions that are not in the keep list
        versions.retain(|v| {
            if keep.iter().any(|k| k.model_id == v.model_id && k.version == v.version) {
                true
            } else {
                removed_versions.push(v.clone());
                false
            }
        });
        
        if !removed_versions.is_empty() {
            self.save_version_index(&versions).await?;
            warn!("Removed {} old versions", removed_versions.len());
        }
        
        Ok(removed_versions)
    }

    /// Calculate checksum for a model
    async fn calculate_model_checksum(&self, model_path: &Path) -> Result<Option<String>> {
        let mut hasher = sha2::Sha256::default();
        let mut total_bytes = 0u64;
        
        let mut entries = tokio_fs::read_dir(model_path).await
            .map_err(|e| ModelError::validation_error(
                model_path.to_string_lossy().as_ref(),
                &format!("Failed to read model directory: {}", e),
                "Check directory permissions"
            ))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ModelError::validation_error(
                model_path.to_string_lossy().as_ref(),
                &format!("Failed to read directory entry: {}", e),
                "Check file system integrity"
            ))? 
        {
            let path = entry.path();
            
            if path.is_file() && !path.file_name().unwrap_or_default().to_string_lossy().starts_with(".") {
                let mut file = tokio_fs::File::open(&path).await
                    .map_err(|e| ModelError::validation_error(
                        path.to_string_lossy().as_ref(),
                        &format!("Failed to open file: {}", e),
                        "Check file permissions"
                    ))?;
                
                let mut buffer = vec![0u8; 8192];
                
                loop {
                    let bytes_read = file.read(&mut buffer).await
                        .map_err(|e| ModelError::validation_error(
                            path.to_string_lossy().as_ref(),
                            &format!("Failed to read file: {}", e),
                            "Check file system integrity"
                        ))?;
                    
                    if bytes_read == 0 {
                        break;
                    }
                    
                    hasher.update(&buffer[..bytes_read]);
                    total_bytes += bytes_read as u64;
                }
            }
        }
        
        if total_bytes > 0 {
            let hash = hex::encode(hasher.finalize());
            Ok(Some(hash))
        } else {
            Ok(None)
        }
    }

    /// Get model statistics
    pub async fn get_statistics(&self) -> Result<ModelVersionStats> {
        let versions = self.load_version_index().await?;
        
        let total_models = versions.iter().map(|v| v.model_id.clone()).collect::<std::collections::HashSet<_>>().len();
        let total_versions = versions.len();
        let total_size: u64 = versions.iter().map(|v| v.size_bytes).sum();
        let pinned_count = versions.iter().filter(|v| v.is_pinned).count();
        
        let oldest_date = versions.iter()
            .map(|v| v.download_date)
            .min();
        
        let newest_date = versions.iter()
            .map(|v| v.download_date)
            .max();
        
        Ok(ModelVersionStats {
            total_models,
            total_versions,
            total_size_bytes: total_size,
            pinned_count,
            oldest_download_date: oldest_date,
            newest_download_date: newest_date,
        })
    }
}

/// Model version statistics
#[derive(Debug, Clone)]
pub struct ModelVersionStats {
    pub total_models: usize,
    pub total_versions: usize,
    pub total_size_bytes: u64,
    pub pinned_count: usize,
    pub oldest_download_date: Option<DateTime<Utc>>,
    pub newest_download_date: Option<DateTime<Utc>>,
}

impl ModelVersionStats {
    /// Format statistics for display
    pub fn format(&self) -> String {
        let size_mb = self.total_size_bytes as f64 / (1024.0 * 1024.0);
        let size_gb = size_mb / 1024.0;
        
        format!(
            "Model Version Statistics:\n\
             - Total models: {}\n\
             - Total versions: {}\n\
             - Total size: {:.2} GB ({:.2} MB)\n\
             - Pinned versions: {}\n\
             - Oldest download: {}\n\
             - Newest download: {}",
            self.total_models,
            self.total_versions,
            size_gb,
            size_mb,
            self.pinned_count,
            self.oldest_download_date.map(|d| d.to_rfc3339()).unwrap_or("N/A".to_string()),
            self.newest_download_date.map(|d| d.to_rfc3339()).unwrap_or("N/A".to_string())
        )
    }
}

/// CLI integration for model version management
pub struct VersionManagerCLI {
    manager: ModelVersionManager,
}

impl VersionManagerCLI {
    /// Create a new version manager CLI
    pub fn new(models_dir: &Path) -> Result<Self> {
        Ok(Self {
            manager: ModelVersionManager::new(models_dir)?,
        })
    }

    /// List all models and their versions
    pub async fn list_versions(&self, model_id: Option<&str>) -> Result<()> {
        let models = if let Some(model_id) = model_id {
            let versions = self.manager.get_model_versions(model_id).await?;
            let mut result = HashMap::new();
            result.insert(model_id.to_string(), versions);
            result
        } else {
            self.manager.list_models().await?
        };

        if models.is_empty() {
            println!("No models found.");
            return Ok(());
        }

        for (model_id, versions) in models {
            println!("\n📦 Model: {}", model_id);
            
            if versions.is_empty() {
                println!("  No versions found.");
                continue;
            }

            for version in versions {
                let status = if version.is_pinned { "📌 PINNED" } else { "  " };
                let size_mb = version.size_bytes as f64 / (1024.0 * 1024.0);
                let date = version.download_date.format("%Y-%m-%d %H:%M");
                
                println!("  {} Version: {} ({:.2} MB)", status, version.version, size_mb);
                println!("      📅 Downloaded: {}", date);
                if let Some(checksum) = &version.checksum {
                    println!("      🔍 Checksum: {}", checksum);
                }
            }
        }

        Ok(())
    }

    /// Pin a version
    pub async fn pin_version(&self, model_id: &str, version: &str) -> Result<()> {
        self.manager.pin_version(model_id, version).await?;
        println!("✅ Pinned version {} for model {}", version, model_id);
        Ok(())
    }

    /// Unpin a version
    pub async fn unpin_version(&self, model_id: &str, version: &str) -> Result<()> {
        self.manager.unpin_version(model_id, version).await?;
        println!("📤 Unpinned version {} for model {}", version, model_id);
        Ok(())
    }

    /// Show version statistics
    pub async fn show_statistics(&self) -> Result<()> {
        let stats = self.manager.get_statistics().await?;
        println!("{}", stats.format());
        Ok(())
    }

    /// Clean up old versions
    pub async fn cleanup_versions(&self, model_id: Option<&str>, keep_latest: usize, keep_pinned: bool) -> Result<()> {
        let removed = self.manager.cleanup_old_versions(model_id, keep_latest, keep_pinned).await?;
        
        if removed.is_empty() {
            println!("🟢 No versions to clean up.");
        } else {
            println!("🗑️  Removed {} old versions:", removed.len());
            for version in removed {
                let size_mb = version.size_bytes as f64 / (1024.0 * 1024.0);
                println!("  - {} {} ({:.2} MB)", version.model_id, version.version, size_mb);
            }
        }
        
        Ok(())
    }

    /// Get the latest version for a model
    pub async fn get_latest(&self, model_id: &str) -> Result<Option<ModelVersion>> {
        Ok(self.manager.get_latest_version(model_id).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_version_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelVersionManager::new(temp_dir.path()).unwrap();
        assert!(manager.versions_dir.exists());
    }

    #[test]
    fn test_model_version_stats_format() {
        let stats = ModelVersionStats {
            total_models: 5,
            total_versions: 12,
            total_size_bytes: 1024 * 1024 * 500, // 500 MB
            pinned_count: 3,
            oldest_download_date: Some(Utc::now() - chrono::Duration::days(30)),
            newest_download_date: Some(Utc::now()),
        };
        
        let formatted = stats.format();
        assert!(formatted.contains("5"));
        assert!(formatted.contains("12"));
        assert!(formatted.contains("500.00 MB"));
        assert!(formatted.contains("3"));
    }

    #[tokio::test]
    async fn test_version_registration() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelVersionManager::new(temp_dir.path()).unwrap();
        
        let model_path = temp_dir.path().join("test_model");
        fs::create_dir_all(&model_path).unwrap();
        fs::write(model_path.join("config.json"), "{}").unwrap();
        
        let version = manager.register_version(
            "test/model", 
            "v1.0", 
            "test-model-v1.0", 
            &model_path
        ).await;
        
        assert!(version.is_ok());
        let version = version.unwrap();
        assert_eq!(version.model_id, "test/model");
        assert_eq!(version.version, "v1.0");
    }
}