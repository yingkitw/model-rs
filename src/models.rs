//! Model management - listing, deploying, and managing local models

use crate::error::{ModelError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;
use std::time::{SystemTime, UNIX_EPOCH};

/// Information about a local model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub file_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    SafeTensors,
    QuantizedSafeTensors { quantization: String },
    GGUF { quantization: String },
    Unknown,
}

const INDEX_FILENAME: &str = ".model_rs_index.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelIndexFile {
    version: u32,
    generated_at_epoch_ms: u64,
    models: Vec<LocalModelInfo>,
}

fn index_path(models_dir: &Path) -> PathBuf {
    models_dir.join(INDEX_FILENAME)
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

fn try_load_index(models_dir: &Path) -> Result<Option<Vec<LocalModelInfo>>> {
    let path = index_path(models_dir);
    if !path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(&path).map_err(|e| {
        ModelError::LocalModelError(format!("Failed to read model index: {}", e))
    })?;
    let file: ModelIndexFile = serde_json::from_str(&raw).map_err(|e| {
        ModelError::LocalModelError(format!("Failed to parse model index: {}", e))
    })?;
    Ok(Some(file.models))
}

fn write_index(models_dir: &Path, models: &[LocalModelInfo]) -> Result<()> {
    let path = index_path(models_dir);
    let payload = ModelIndexFile {
        version: 1,
        generated_at_epoch_ms: now_epoch_ms(),
        models: models.to_vec(),
    };

    let raw = serde_json::to_vec_pretty(&payload).map_err(|e| {
        ModelError::LocalModelError(format!("Failed to serialize model index: {}", e))
    })?;
    fs::write(&path, raw).map_err(|e| {
        ModelError::LocalModelError(format!("Failed to write model index: {}", e))
    })?;
    Ok(())
}

/// List all models in the models directory
pub fn list_models(models_dir: Option<&Path>) -> Result<Vec<LocalModelInfo>> {
    let search_path = models_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./models"));

    info!("Listing models in: {}", search_path.display());

    if !search_path.exists() {
        return Ok(vec![]);
    }

    // Fast path: use cached index if available.
    if let Some(models) = try_load_index(&search_path)? {
        return Ok(models);
    }

    let models = list_models_scan(&search_path)?;
    // Best-effort cache write (don’t fail `list` if indexing fails).
    let _ = write_index(&search_path, &models);
    Ok(models)
}

/// Refresh (rebuild) the on-disk model index used by `list_models`.
pub fn refresh_models_index(models_dir: Option<&Path>) -> Result<()> {
    let models_path = models_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./models"));

    if !models_path.exists() {
        // Nothing to index yet.
        return Ok(());
    }

    let models = list_models_scan(&models_path)?;
    write_index(&models_path, &models)
}

fn list_models_scan(models_dir: &Path) -> Result<Vec<LocalModelInfo>> {
    let mut models = Vec::new();

    let entries = fs::read_dir(models_dir).map_err(|e| {
        ModelError::ModelNotFound(format!("Failed to read models directory: {}", e))
    })?;

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip if not a directory
        if !path.is_dir() {
            continue;
        }

        // Check for model files
        let model_info = analyze_model_directory(&path)?;
        if let Some(info) = model_info {
            models.push(info);
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

/// Analyze a model directory to determine its format and metadata
fn analyze_model_directory(path: &Path) -> Result<Option<LocalModelInfo>> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Unknown")
        .to_string();

    let mut size_bytes = 0u64;
    let mut file_count = 0usize;
    let mut has_safetensors = false;
    let mut has_gguf = false;
    let mut safetensors_quantization: Option<String> = None;
    let mut gguf_quantization = None;
    let mut architecture = String::from("Unknown");

    // Read directory contents
    let entries = fs::read_dir(path)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to read directory: {}", e)))?;

    for entry in entries.flatten() {
        let file_path = entry.path();

        // Skip directories
        if file_path.is_dir() {
            continue;
        }

        // Get metadata
        if let Ok(metadata) = file_path.metadata() {
            size_bytes += metadata.len();
            file_count += 1;
        }

        // Check for GGUF files
        if let Some(ext) = file_path.extension() {
            if ext == "gguf" {
                has_gguf = true;
                // Try to detect quantization from filename
                if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
                    let filename_lower = filename.to_lowercase();
                    if filename_lower.contains("q2_k") {
                        gguf_quantization = Some("Q2_K");
                    } else if filename_lower.contains("q4_k_m") {
                        gguf_quantization = Some("Q4_K_M");
                    } else if filename_lower.contains("q4_k") {
                        gguf_quantization = Some("Q4_K");
                    } else if filename_lower.contains("q5_k_m") {
                        gguf_quantization = Some("Q5_K_M");
                    } else if filename_lower.contains("q5_k") {
                        gguf_quantization = Some("Q5_K");
                    } else if filename_lower.contains("q6_k") {
                        gguf_quantization = Some("Q6_K");
                    } else if filename_lower.contains("q8_0") {
                        gguf_quantization = Some("Q8_0");
                    } else if filename_lower.contains("f16") {
                        gguf_quantization = Some("F16");
                    }
                }
            } else if ext == "safetensors" {
                has_safetensors = true;
                // Try to detect common non-GGUF quantization formats by filename.
                if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
                    let filename_lower = filename.to_lowercase();
                    if filename_lower.contains("int8") || filename_lower.contains("8bit") {
                        safetensors_quantization = Some("INT8".to_string());
                    } else if filename_lower.contains("int4") || filename_lower.contains("4bit") {
                        safetensors_quantization = Some("INT4".to_string());
                    } else if filename_lower.contains("awq") {
                        safetensors_quantization = Some("AWQ".to_string());
                    } else if filename_lower.contains("gptq") {
                        safetensors_quantization = Some("GPTQ".to_string());
                    } else if filename_lower.contains("bnb") || filename_lower.contains("bitsandbytes") {
                        safetensors_quantization = Some("BITSANDBYTES".to_string());
                    } else if filename_lower.contains("q4_") {
                        safetensors_quantization = Some("Q4".to_string());
                    } else if filename_lower.contains("q8_") {
                        safetensors_quantization = Some("Q8".to_string());
                    }
                }
            }
        }

        // Try to read architecture from config.json
        if file_path.file_name() == Some(std::ffi::OsStr::new("config.json")) {
            if let Ok(content) = fs::read_to_string(&file_path) {
                if let Ok(config) = serde_json::from_str::<Value>(&content) {
                    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                        architecture = model_type.to_string();
                    }
                }
            }
        }
    }

    // Determine model format
    let format = if has_gguf {
        ModelFormat::GGUF {
            quantization: gguf_quantization.unwrap_or("Unknown").to_string(),
        }
    } else if has_safetensors {
        match safetensors_quantization {
            Some(q) => ModelFormat::QuantizedSafeTensors { quantization: q },
            None => ModelFormat::SafeTensors,
        }
    } else {
        ModelFormat::Unknown
    };

    // Only include if it has model files
    if matches!(format, ModelFormat::Unknown) {
        return Ok(None);
    }

    Ok(Some(LocalModelInfo {
        name,
        path: path.to_path_buf(),
        architecture,
        format,
        size_bytes,
        file_count,
    }))
}

/// Display model information in a formatted table
pub fn display_models(models: &[LocalModelInfo], formatter: &crate::output::OutputFormatter) {
    if models.is_empty() {
        formatter.print_warning("No models found in the models directory.");
        formatter.print_markdown("\n**To download a model:**\n\n```bash\nmodel-rs download <model-name>\n```\n\n**Example:**\n\n```bash\nmodel-rs download TinyLlama/TinyLlama-1.1B-Chat-v1.0\n```\n");
        return;
    }

    formatter.print_header("Local Models");

    for model in models.iter() {
        let format_str = match &model.format {
            ModelFormat::SafeTensors => "SafeTensors".to_string(),
            ModelFormat::QuantizedSafeTensors { quantization } => {
                format!("SafeTensors ({})", quantization)
            }
            ModelFormat::GGUF { quantization } => format!("GGUF ({})", quantization),
            ModelFormat::Unknown => "Unknown".to_string(),
        };

        let size_mb = model.size_bytes / (1024 * 1024);
        let size_gb = size_mb / 1024;

        let size_str = if size_gb > 0 {
            format!("{} GB", size_gb)
        } else {
            format!("{} MB", size_mb)
        };

        formatter.print_model_info(
            &model.name,
            &model.path.display().to_string(),
            &format_str,
            &model.architecture,
            &size_str,
            model.file_count,
        );
    }

    formatter.print_info(&format!("Total: {} model(s)", models.len()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_list_models_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let models = list_models(Some(tmp.path())).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_list_models_nonexistent_directory() {
        let models = list_models(Some(Path::new("/nonexistent/path"))).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_analyze_model_directory_with_safetensors() {
        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("test-model");
        fs::create_dir(&model_dir).unwrap();

        // Create fake safetensors file
        fs::write(model_dir.join("model.safetensors"), b"fake content").unwrap();

        // Create config.json
        fs::write(model_dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();

        let info = analyze_model_directory(&model_dir).unwrap();
        assert!(info.is_some());

        let model_info = info.unwrap();
        assert_eq!(model_info.name, "test-model");
        assert!(matches!(model_info.format, ModelFormat::SafeTensors));
        assert_eq!(model_info.architecture, "llama");
    }

    #[test]
    fn test_analyze_model_directory_with_quantized_safetensors() {
        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("test-model");
        fs::create_dir(&model_dir).unwrap();

        // Create fake quantized safetensors file name
        fs::write(model_dir.join("model-int4-4bit.safetensors"), b"fake content").unwrap();

        // Create config.json
        fs::write(model_dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();

        let info = analyze_model_directory(&model_dir).unwrap();
        assert!(info.is_some());

        let model_info = info.unwrap();
        assert_eq!(model_info.name, "test-model");
        assert!(matches!(
            model_info.format,
            ModelFormat::QuantizedSafeTensors { .. }
        ));

        if let ModelFormat::QuantizedSafeTensors { quantization } = model_info.format {
            assert_eq!(quantization, "INT4");
        }
        assert_eq!(model_info.architecture, "llama");
    }

    #[test]
    fn test_analyze_model_directory_with_gguf() {
        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("test-model");
        fs::create_dir(&model_dir).unwrap();

        // Create fake GGUF file
        fs::write(model_dir.join("model-q4_k_m.gguf"), b"fake content").unwrap();

        let info = analyze_model_directory(&model_dir).unwrap();
        assert!(info.is_some());

        let model_info = info.unwrap();
        assert_eq!(model_info.name, "test-model");
        assert!(matches!(model_info.format, ModelFormat::GGUF { .. }));

        if let ModelFormat::GGUF { quantization } = model_info.format {
            assert_eq!(quantization, "Q4_K_M");
        }
    }
}
