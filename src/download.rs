use crate::error::{ModelError, Result};
use crate::config::DEFAULT_MIRROR;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
struct RepoFile {
    path: String,
    #[serde(rename = "type")]
    file_type: String,
}

pub async fn download_model(
    model: &str,
    mirror: Option<&str>,
    output: Option<&Path>,
) -> Result<()> {
    let mirror_url = mirror.unwrap_or(DEFAULT_MIRROR);
    let output_dir = get_output_dir(model, output)?;

    info!("Downloading model '{}' from {}", model, mirror_url);
    info!("Output directory: {}", output_dir.display());

    fs::create_dir_all(&output_dir).await?;

    let client = Client::builder()
        .user_agent("model-rs/0.1.0")
        .build()?;

    // Check if model exists before downloading
    check_model_exists(&client, mirror_url, model).await?;

    // Fetch the list of files to download (with fallback)
    let files_to_download = get_model_files(&client, mirror_url, model).await?;

    let total_files = files_to_download.len();
    info!("Starting download of {} files...", total_files);

    let overall_progress = ProgressBar::new(total_files as u64);
    overall_progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%)")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut skipped_files: Vec<String> = Vec::new();
    let mut successful_downloads: usize = 0;

    for file in &files_to_download {
        let url = format!("{}/{}/resolve/main/{}", mirror_url, model, file);
        let file_path = output_dir.join(file);

        // Validate the file path is within the output directory (prevent path traversal)
        let canonical_file = file_path
            .canonicalize()
            .unwrap_or_else(|_| file_path.clone());
        let canonical_output = output_dir
            .canonicalize()
            .unwrap_or_else(|_| output_dir.clone());

        if !canonical_file.starts_with(&canonical_output) {
            warn!("Skipping file with invalid path: {}", file);
            continue;
        }

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        info!("Downloading: {}", file);

        match download_file(&client, &url, &file_path).await {
            Ok(()) => {
                successful_downloads += 1;
            }
            Err(e) => {
                // Check if this is a 403 Forbidden error (gated model)
                if e.to_string().contains("403") {
                    warn!("Skipping {} (access forbidden - this may be a gated model file)", file);
                    skipped_files.push(file.clone());
                } else {
                    // For other errors, still report but continue
                    warn!("Failed to download {}: {}", file, e);
                    skipped_files.push(file.clone());
                }
            }
        }

        overall_progress.inc(1);
    }

    overall_progress.finish_with_message(format!("Download complete: {}/{} files", successful_downloads, total_files));

    if !skipped_files.is_empty() {
        warn!("Skipped files: {}", skipped_files.join(", "));
        warn!("Some files could not be downloaded. This model may be gated or require authentication.");
        warn!("Visit https://huggingface.co/{}/request-access to request access if needed.", model);
    }

    if successful_downloads == 0 {
        return Err(ModelError::DownloadError(
            format!("No files could be downloaded. The model '{}' may be gated or require authentication. Visit https://huggingface.co/{}/request-access to request access.", model, model)
        ));
    }

    info!("Model downloaded successfully to: {}", output_dir.display());

    // Validate the downloaded model
    info!("Validating downloaded model files...");
    validate_model(&output_dir).await?;

    // Refresh the cached local model index (best-effort).
    if let Some(parent) = output_dir.parent() {
        let _ = crate::models::refresh_models_index(Some(parent));
    }

    Ok(())
}

fn get_output_dir(model: &str, output: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = output {
        // Validate the custom output path to prevent directory traversal issues
        let canonical_path = path
            .canonicalize()
            .map_err(|e| ModelError::DownloadError(format!("Invalid output path '{}': {}", path.display(), e)))?;
        return Ok(canonical_path);
    }

    // Use a local working folder "models" instead of system directory
    // Sanitize model name to prevent directory traversal
    let model_name = model
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>();
    Ok(PathBuf::from("models").join(model_name))
}

async fn check_model_exists(client: &Client, mirror_url: &str, model: &str) -> Result<()> {
    // Try to fetch the model info to check if it exists
    let info_url = format!("{}/api/models/{}", mirror_url, model);

    let response = client
        .head(&info_url)
        .send()
        .await
        .map_err(|e| ModelError::DownloadError(format!("Failed to check model availability: {}", e)))?;

    if response.status().is_success() {
        return Ok(());
    }

    // Also try checking if config.json exists (some mirrors don't support the API endpoint)
    let config_url = format!("{}/{}/resolve/main/config.json", mirror_url, model);
    let response = client
        .head(&config_url)
        .send()
        .await
        .map_err(|e| ModelError::DownloadError(format!("Failed to check model availability: {}", e)))?;

    if !response.status().is_success() {
        return Err(ModelError::DownloadError(format!(
            "Model '{}' not found. Please verify the model name.\n\nHint: Model names should be in the format 'org/model-name' (e.g., 'bert-base-uncased', 'google/flan-t5-small').\nYou can search for models at https://hf-mirror.com or https://huggingface.co/models",
            model
        )));
    }

    Ok(())
}

/// Fetches the list of files for a model from the HuggingFace API
async fn fetch_model_files(client: &Client, mirror_url: &str, model: &str) -> Result<Vec<String>> {
    let tree_url = format!("{}/api/models/{}/tree/main", mirror_url, model);

    info!("Fetching file list from HuggingFace API...");

    let response = client
        .get(&tree_url)
        .send()
        .await
        .map_err(|e| ModelError::DownloadError(format!("Failed to fetch model files: {}", e)))?;

    if !response.status().is_success() {
        return Err(ModelError::DownloadError(format!(
            "HTTP {} when fetching file list from {}",
            response.status(),
            tree_url
        )));
    }

    let files: Vec<RepoFile> = response
        .json()
        .await
        .map_err(|e| ModelError::DownloadError(format!("Failed to parse file list: {}", e)))?;

    // Filter to only files (not directories), excluding hidden files
    let model_files: Vec<String> = files
        .into_iter()
        .filter(|f| f.file_type == "file" && !f.path.starts_with('.'))
        .map(|f| f.path)
        .collect();

    if model_files.is_empty() {
        return Err(ModelError::DownloadError(
            "No files found for this model".to_string(),
        ));
    }

    info!("Found {} files to download", model_files.len());
    Ok(model_files)
}

/// Gets the list of files to download, using dynamic discovery with fallback
async fn get_model_files(client: &Client, mirror_url: &str, model: &str) -> Result<Vec<String>> {
    // Try dynamic discovery first
    match fetch_model_files(client, mirror_url, model).await {
        Ok(files) => Ok(files),
        Err(e) => {
            info!("Dynamic file discovery failed: {}", e);
            info!("Falling back to hardcoded file list for this model");

            // Fallback to hardcoded lists
            if model.contains("granite") {
                Ok(vec![
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "special_tokens_map.json".to_string(),
                    "model.safetensors".to_string(),
                ])
            } else {
                Ok(vec![
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "pytorch_model.bin".to_string(),
                ])
            }
        }
    }
}

async fn download_file(client: &Client, url: &str, path: &Path) -> Result<()> {
    const MAX_RETRIES: u32 = 3;
    let mut attempt = 0;

    loop {
        attempt += 1;

        match download_file_attempt(client, url, path).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                // Don't retry on 404 or 403 (client errors)
                if e.to_string().contains("404") || e.to_string().contains("403") {
                    return Err(e);
                }

                if attempt >= MAX_RETRIES {
                    return Err(ModelError::DownloadError(format!(
                        "Failed after {} attempts: {}",
                        MAX_RETRIES, e
                    )));
                }

                let delay = 2_u64.pow(attempt - 1);
                warn!("Download attempt {} failed: {}. Retrying in {}s...", attempt, e, delay);
                tokio::time::sleep(tokio::time::Duration::from_secs(delay)).await;
            }
        }
    }
}

async fn download_file_attempt(client: &Client, url: &str, path: &Path) -> Result<()> {
    // Check if file exists and get its size for resume capability
    let existing_size = if path.exists() {
        let metadata = fs::metadata(path).await?;
        Some(metadata.len())
    } else {
        None
    };

    // Build request with Range header if resuming
    let response = if let Some(size) = existing_size {
        info!("Resuming download from {} bytes", size);
        client
            .get(url)
            .header("Range", format!("bytes={}-", size))
            .send()
            .await?
    } else {
        client.get(url).send().await?
    };

    // Check for partial content response (206) when resuming
    let is_resuming = response.status() == 206;
    if !response.status().is_success() && !is_resuming {
        return Err(ModelError::DownloadError(format!(
            "HTTP {}: {}",
            response.status(),
            url
        )));
    }

    let total_size = response.content_length().unwrap_or(0);
    let downloaded_so_far = existing_size.unwrap_or(0);
    let total_size_with_resume = if is_resuming { total_size + downloaded_so_far } else { total_size };

    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");

    let pb = if total_size_with_resume > 0 {
        let pb = ProgressBar::new(total_size_with_resume);
        pb.set_position(downloaded_so_far);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})\n   {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        if existing_size.is_some() {
            pb.set_message(format!("Resuming: {}", file_name));
        } else {
            pb.set_message(format!("Downloading: {}", file_name));
        }
        Some(pb)
    } else {
        // For files without known size, show a spinner
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        pb.set_message(format!("{}: {} (size unknown)",
            if existing_size.is_some() { "Resuming" } else { "Downloading" },
            file_name
        ));
        Some(pb)
    };

    // Open file in append mode if resuming, create mode if new
    let mut file = if existing_size.is_some() {
        File::options().append(true).open(path).await?
    } else {
        File::create(path).await?
    };

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = downloaded_so_far;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if let Some(ref pb) = pb {
            pb.set_position(downloaded);

            // Update message periodically with progress percentage
            if total_size_with_resume > 0 {
                let percent = (downloaded as f64 / total_size_with_resume as f64 * 100.0) as u64;
                pb.set_message(format!("{}: {}%", file_name, percent));
            }
        }
    }

    if let Some(pb) = pb {
        let status = if existing_size.is_some() { "Resumed OK" } else { "OK" };
        pb.finish_with_message(format!("{} {}", status, file_name));
    }

    file.sync_all().await?;

    // Validate the downloaded file size matches expected size
    if total_size > 0 {
        let final_size = fs::metadata(path).await?.len();
        let expected_total = if is_resuming { total_size + downloaded_so_far } else { total_size };
        if is_resuming && final_size != expected_total {
            warn!("Downloaded file size ({}) doesn't match expected size ({}) after resume",
                final_size, expected_total);
        }
    }

    Ok(())
}

/// Validates that the downloaded model contains all required files
async fn validate_model(output_dir: &Path) -> Result<()> {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Check for essential files
    let essential_files = vec!["config.json"];
    for file in &essential_files {
        let path = output_dir.join(file);
        if !path.exists() {
            errors.push(format!("Missing essential file: {}", file));
        }
    }

    // Check for tokenizer files (at least one should exist)
    let tokenizer_files = vec!["tokenizer.json", "tokenizer_config.json"];
    let has_tokenizer = tokenizer_files.iter()
        .any(|file| output_dir.join(file).exists());

    if !has_tokenizer {
        errors.push("Missing tokenizer files (tokenizer.json or tokenizer_config.json required)".to_string());
    }

    // Check for model weight files
    let has_safetensors = find_files_with_extension(output_dir, ".safetensors").await?;
    let has_bin = find_files_with_extension(output_dir, ".bin").await?;

    if !has_safetensors && !has_bin {
        warnings.push("No model weight files found (.safetensors or .bin). Model inference will not work.".to_string());
    }

    // Validate config.json if it exists
    let config_path = output_dir.join("config.json");
    if config_path.exists() {
        match validate_config_json(&config_path).await {
            Ok(_) => info!("config.json is valid"),
            Err(e) => warnings.push(format!("config.json validation warning: {}", e)),
        }
    }

    // Report errors
    if !errors.is_empty() {
        return Err(ModelError::DownloadError(format!(
            "Model validation failed:\n  {}",
            errors.join("\n  ")
        )));
    }

    // Report warnings
    if !warnings.is_empty() {
        for warning in &warnings {
            warn!("{}", warning);
        }
    }

    info!("Model validation complete!");
    Ok(())
}

/// Finds files with a specific extension in the directory
async fn find_files_with_extension(dir: &Path, extension: &str) -> Result<bool> {
    let mut entries = fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy() == extension.trim_start_matches('.') {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Validates that config.json can be parsed and contains required fields
async fn validate_config_json(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path).await
        .map_err(|e| ModelError::DownloadError(format!("Failed to read config.json: {}", e)))?;

    let _config: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| ModelError::DownloadError(format!("Failed to parse config.json: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use std::io::Write;

    #[tokio::test]
    async fn test_get_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output = get_output_dir("ibm/granite", Some(temp_dir.path())).unwrap();
        // Compare canonicalized paths since get_output_dir now canonicalizes
        let canonical_input = temp_dir.path().canonicalize().unwrap();
        assert_eq!(output, canonical_input);
    }

    #[tokio::test]
    async fn test_validate_model_with_all_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create essential files
        let config_path = temp_dir.path().join("config.json");
        let mut file = fs::File::create(&config_path).unwrap();
        writeln!(file, r#"{{"model_type": "llama", "hidden_size": 768}}"#).unwrap();

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = fs::File::create(&tokenizer_path).unwrap();
        writeln!(file, r#"{{"vocab": {{}}"}}"#).unwrap();

        let model_path = temp_dir.path().join("model.safetensors");
        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        // Should validate successfully
        let result = validate_model(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_model_missing_config() {
        let temp_dir = TempDir::new().unwrap();

        // Only create tokenizer, missing config
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = fs::File::create(&tokenizer_path).unwrap();
        writeln!(file, r#"{{"vocab": {{}}"}}"#).unwrap();

        // Should fail with missing config error
        let result = validate_model(temp_dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing essential file: config.json"));
    }

    #[tokio::test]
    async fn test_validate_model_missing_tokenizer() {
        let temp_dir = TempDir::new().unwrap();

        // Only create config, missing tokenizer
        let config_path = temp_dir.path().join("config.json");
        let mut file = fs::File::create(&config_path).unwrap();
        writeln!(file, r#"{{"model_type": "llama"}}"#).unwrap();

        // Should fail with missing tokenizer error
        let result = validate_model(temp_dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing tokenizer files"));
    }

    #[tokio::test]
    async fn test_validate_model_no_weights() {
        let temp_dir = TempDir::new().unwrap();

        // Create config and tokenizer, but no model weights
        let config_path = temp_dir.path().join("config.json");
        let mut file = fs::File::create(&config_path).unwrap();
        writeln!(file, r#"{{"model_type": "llama"}}"#).unwrap();

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = fs::File::create(&tokenizer_path).unwrap();
        writeln!(file, r#"{{"vocab": {{}}"}}"#).unwrap();

        // Should validate but with warning (no error)
        let result = validate_model(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_model_invalid_config_json() {
        let temp_dir = TempDir::new().unwrap();

        // Create invalid config.json
        let config_path = temp_dir.path().join("config.json");
        let mut file = fs::File::create(&config_path).unwrap();
        writeln!(file, "invalid json {{{{").unwrap();

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = fs::File::create(&tokenizer_path).unwrap();
        writeln!(file, r#"{{"vocab": {{}}"}}"#).unwrap();

        // Should validate but with warning about invalid config
        let result = validate_model(temp_dir.path()).await;
        assert!(result.is_ok()); // Validation passes but with warning
    }

    #[tokio::test]
    async fn test_find_files_with_extension_safetensors() {
        let temp_dir = TempDir::new().unwrap();

        // Create a .safetensors file
        let model_path = temp_dir.path().join("model.safetensors");
        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        let result = find_files_with_extension(temp_dir.path(), ".safetensors").await;
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[tokio::test]
    async fn test_find_files_with_extension_bin() {
        let temp_dir = TempDir::new().unwrap();

        // Create a .bin file
        let model_path = temp_dir.path().join("pytorch_model.bin");
        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        let result = find_files_with_extension(temp_dir.path(), ".bin").await;
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[tokio::test]
    async fn test_find_files_with_extension_not_found() {
        let temp_dir = TempDir::new().unwrap();

        let result = find_files_with_extension(temp_dir.path(), ".safetensors").await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[tokio::test]
    async fn test_validate_config_json_valid() {
        let temp_dir = TempDir::new().unwrap();

        let config_path = temp_dir.path().join("config.json");
        let mut file = fs::File::create(&config_path).unwrap();
        writeln!(file, r#"{{"model_type": "llama", "hidden_size": 768, "num_layers": 12}}"#).unwrap();

        let result = validate_config_json(&config_path).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_config_json_invalid() {
        let temp_dir = TempDir::new().unwrap();

        let config_path = temp_dir.path().join("config.json");
        let mut file = fs::File::create(&config_path).unwrap();
        writeln!(file, "invalid json {{{{").unwrap();

        let result = validate_config_json(&config_path).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse config.json"));
    }
}
