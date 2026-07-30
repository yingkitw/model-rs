//! Model integrity verification module for model-rs
//!
//! This module provides functionality to verify the integrity of downloaded models
//! using checksums and cryptographic hashes to ensure data integrity and prevent tampering.

use crate::error::{ModelError, Result};
use crate::validation::*;
use sha2::{Sha256, Digest};
use hex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::fs as tokio_fs;
use tokio::io::AsyncReadExt;
use tracing::{info, warn, debug};

/// Supported checksum algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ChecksumAlgorithm {
    Sha256,
    Sha1,
    Md5,
}

/// Checksum information for a model file
#[derive(Debug, Clone)]
pub struct FileChecksum {
    pub path: PathBuf,
    pub algorithm: ChecksumAlgorithm,
    pub expected_hash: String,
    pub actual_hash: String,
    pub is_valid: bool,
}

/// Model integrity verifier
pub struct ModelVerifier {
    model_path: PathBuf,
    checksums: HashMap<String, String>,
}

impl ModelVerifier {
    /// Create a new model verifier for the given model path
    pub async fn new(model_path: &Path) -> Result<Self> {
        validate_model_directory(model_path).await?;

        let checksums = Self::load_checksums(model_path)?;
        
        Ok(Self {
            model_path: model_path.to_path_buf(),
            checksums,
        })
    }

    /// Load checksum information from model directory
    fn load_checksums(model_path: &Path) -> Result<HashMap<String, String>> {
        let checksum_file = model_path.join(".model_checksums");
        
        if !checksum_file.exists() {
            // If no checksum file exists, we can't verify, but we can create one
            warn!("No checksum file found at {}", checksum_file.display());
            return Ok(HashMap::new());
        }

        let content = fs::read_to_string(&checksum_file)
            .map_err(|e| ModelError::validation_error(
                checksum_file.to_string_lossy().as_ref(),
                &format!("Failed to read checksum file: {}", e),
                "Check file permissions and integrity"
            ))?;

        let checksums: HashMap<String, String> = serde_json::from_str(&content)
            .map_err(|e| ModelError::validation_error(
                checksum_file.to_string_lossy().as_ref(),
                &format!("Invalid checksum file format: {}", e),
                "Check the JSON format of the checksum file"
            ))?;

        Ok(checksums)
    }

    /// Save checksum information for the model directory
    pub async fn save_checksums(&self) -> Result<()> {
        let checksum_file = self.model_path.join(".model_checksums");
        
        let checksums = self.calculate_checksums().await?;
        
        let content = serde_json::to_string_pretty(&checksums)
            .map_err(|e| ModelError::validation_error(
                checksum_file.to_string_lossy().as_ref(),
                &format!("Failed to serialize checksums: {}", e),
                "Check the model directory contents"
            ))?;

        tokio_fs::write(&checksum_file, content)
            .await
            .map_err(|e| ModelError::validation_error(
                checksum_file.to_string_lossy().as_ref(),
                &format!("Failed to write checksum file: {}", e),
                "Check file permissions and disk space"
            ))?;

        info!("Checksums saved to {}", checksum_file.display());
        Ok(())
    }

    /// Calculate checksums for all files in the model directory
    pub async fn calculate_checksums(&self) -> Result<HashMap<String, String>> {
        let mut checksums = HashMap::new();
        let algorithm = ChecksumAlgorithm::Sha256; // Default algorithm

        // Walk through all files in the model directory
        let mut entries = tokio_fs::read_dir(&self.model_path).await
            .map_err(|e| ModelError::validation_error(
                self.model_path.to_string_lossy().as_ref(),
                &format!("Failed to read model directory: {}", e),
                "Check directory permissions"
            ))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ModelError::validation_error(
                self.model_path.to_string_lossy().as_ref(),
                &format!("Failed to read directory entry: {}", e),
                "Check file system integrity"
            ))? 
        {
            let path = entry.path();
            
            if path.is_file() && !path.file_name().unwrap_or_default().to_string_lossy().starts_with(".model_checksums") {
                // Skip checksum files and hidden files (except .model_checksums)
                debug!("Calculating checksum for {}", path.display());
                
                let hash = self.calculate_file_checksum(&path, &algorithm).await?;
                let relative_path = path.strip_prefix(&self.model_path)
                    .map_err(|_| ModelError::validation_error(
                        path.to_string_lossy().as_ref(),
                        "Path is not within model directory",
                        "This should not happen - internal error"
                    ))?;
                
                checksums.insert(relative_path.to_string_lossy().to_string(), hash);
            }
        }

        Ok(checksums)
    }

    /// Calculate checksum for a single file
    async fn calculate_file_checksum(&self, file_path: &Path, algorithm: &ChecksumAlgorithm) -> Result<String> {
        let mut file = tokio_fs::File::open(file_path).await
            .map_err(|e| ModelError::validation_error(
                file_path.to_string_lossy().as_ref(),
                &format!("Failed to open file: {}", e),
                "Check file permissions"
            ))?;

        match algorithm {
            ChecksumAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                let mut buffer = vec![0u8; 8192]; // 8KB chunks
                
                loop {
                    let bytes_read = file.read(&mut buffer).await
                        .map_err(|e| ModelError::validation_error(
                            file_path.to_string_lossy().as_ref(),
                            &format!("Failed to read file: {}", e),
                            "Check file system integrity"
                        ))?;
                    
                    if bytes_read == 0 {
                        break;
                    }
                    
                    hasher.update(&buffer[..bytes_read]);
                }
                
                let result = hasher.finalize();
                Ok(hex::encode(result))
            }
            ChecksumAlgorithm::Sha1 => {
                // Implement SHA1 if needed
                Err(ModelError::validation_error(
                    file_path.to_string_lossy().as_ref(),
                    "SHA1 algorithm not yet implemented",
                    "Use SHA256 instead"
                ))
            }
            ChecksumAlgorithm::Md5 => {
                // Implement MD5 if needed
                Err(ModelError::validation_error(
                    file_path.to_string_lossy().as_ref(),
                    "MD5 algorithm not yet implemented", 
                    "Use SHA256 instead"
                ))
            }
        }
    }

    /// Verify model integrity against stored checksums
    pub async fn verify_model(&self) -> Result<Vec<FileChecksum>> {
        let current_checksums = self.calculate_checksums().await?;
        
        let mut results = Vec::new();
        
        for (file_path, expected_hash) in &self.checksums {
            let full_path = self.model_path.join(file_path);
            
            // Calculate current hash
            let actual_hash = match self.calculate_file_checksum(&full_path, &ChecksumAlgorithm::Sha256).await {
                Ok(hash) => hash,
                Err(e) => {
                    results.push(FileChecksum {
                        path: full_path,
                        algorithm: ChecksumAlgorithm::Sha256,
                        expected_hash: expected_hash.clone(),
                        actual_hash: format!("ERROR: {}", e.to_string()),
                        is_valid: false,
                    });
                    continue;
                }
            };
            
            let is_valid = actual_hash == *expected_hash;
            
            results.push(FileChecksum {
                path: full_path,
                algorithm: ChecksumAlgorithm::Sha256,
                expected_hash: expected_hash.clone(),
                actual_hash,
                is_valid,
            });
        }

        // Check for files that exist but aren't in the checksum file
        for (file_path, current_hash) in &current_checksums {
            if !self.checksums.contains_key(file_path) {
                let full_path = self.model_path.join(file_path);
                results.push(FileChecksum {
                    path: full_path,
                    algorithm: ChecksumAlgorithm::Sha256,
                    expected_hash: "NO CHECKSUM STORED".to_string(),
                    actual_hash: current_hash.clone(),
                    is_valid: false,
                });
            }
        }

        Ok(results)
    }

    /// Get verification summary
    pub fn get_summary(&self, results: &[FileChecksum]) -> ModelVerificationSummary {
        let total_files = results.len();
        let valid_files = results.iter().filter(|r| r.is_valid).count();
        let invalid_files = total_files - valid_files;

        ModelVerificationSummary {
            model_path: self.model_path.clone(),
            total_files,
            valid_files,
            invalid_files,
            is_valid: invalid_files == 0,
        }
    }
}

/// Summary of model verification results
#[derive(Debug, Clone)]
pub struct ModelVerificationSummary {
    pub model_path: PathBuf,
    pub total_files: usize,
    pub valid_files: usize,
    pub invalid_files: usize,
    pub is_valid: bool,
}

impl ModelVerificationSummary {
    /// Create a summary report
    pub fn to_string(&self) -> String {
        let status = if self.is_valid {
            "✓ VERIFIED".to_string()
        } else {
            "✗ VERIFICATION FAILED".to_string()
        };

        format!(
            "{} - Model: {}\n  Total files: {}\n  Valid files: {}\n  Invalid files: {}\n  Status: {}",
            if self.is_valid { "✓" } else { "✗" },
            self.model_path.display(),
            self.total_files,
            self.valid_files,
            self.invalid_files,
            status
        )
    }
}

/// Model integrity manager for handling verification operations
pub struct ModelIntegrityManager {
    models_dir: PathBuf,
}

impl ModelIntegrityManager {
    /// Create a new model integrity manager
    pub fn new(models_dir: &Path) -> Result<Self> {
        validate_path(models_dir, "models directory")?;
        
        Ok(Self {
            models_dir: models_dir.to_path_buf(),
        })
    }

    /// Verify a specific model
    pub async fn verify_model(&self, model_name: &str) -> Result<ModelVerificationSummary> {
        let model_path = self.resolve_model_path(model_name)?;
        let verifier = ModelVerifier::new(&model_path).await?;
        let results = verifier.verify_model().await?;
        let summary = verifier.get_summary(&results);
        
        Ok(summary)
    }

    /// Verify all models in the models directory
    pub async fn verify_all_models(&self) -> Result<Vec<ModelVerificationSummary>> {
        let mut summaries = Vec::new();
        
        if let Ok(mut entries) = tokio_fs::read_dir(&self.models_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                
                if path.is_dir() {
                    let model_name = path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("");
                    
                    // Convert from org--model format to org/model for display
                    let display_name = model_name.replace("--", "/");
                    
                    if let Ok(verifier) = ModelVerifier::new(&path).await {
                        if let Ok(results) = verifier.verify_model().await {
                            let mut summary = verifier.get_summary(&results);
                            summary.model_path = path; // Use the actual path for display
                            summaries.push(summary);
                        }
                    }
                }
            }
        }

        Ok(summaries)
    }

    /// Generate checksums for a model (will overwrite existing)
    pub async fn generate_checksums(&self, model_name: &str) -> Result<()> {
        let model_path = self.resolve_model_path(model_name)?;
        let mut verifier = ModelVerifier::new(&model_path).await?;
        verifier.save_checksums().await?;
        Ok(())
    }

    /// Resolve model name to path
    fn resolve_model_path(&self, model_name: &str) -> Result<PathBuf> {
        validate_model_name(model_name)?;
        
        // Convert org/model format to org--model for directory name
        let dir_name = model_name.replace('/', "--");
        let model_path = self.models_dir.join(dir_name);
        
        if !model_path.exists() {
            return Err(ModelError::model_not_found(
                format!("Model '{}' not found. Download the model first using 'model-rs download <model>'", model_name),
            ));
        }
        
        Ok(model_path)
    }
}

/// Convenience function to verify a model
pub async fn verify_model(model_path: &Path) -> Result<Vec<FileChecksum>> {
    let verifier = ModelVerifier::new(model_path).await?;
    verifier.verify_model().await
}

/// Convenience function to generate checksums for a model
pub async fn generate_model_checksums(model_path: &Path) -> Result<()> {
    let mut verifier = ModelVerifier::new(model_path).await?;
    verifier.save_checksums().await
}

/// CLI integration for model verification
pub async fn cli_verify_model(model_name: Option<&str>, models_dir: Option<&Path>) -> Result<()> {
    let resolved_dir = match models_dir {
        Some(dir) => dir.to_path_buf(),
        None => crate::model_ops::ModelOperations::get_models_dir()?,
    };
    
    let manager = ModelIntegrityManager::new(&resolved_dir)?;
    
    match model_name {
        Some(name) => {
            let summary = manager.verify_model(name).await?;
            println!("{}", summary.to_string());
            
            if !summary.is_valid {
                std::process::exit(1);
            }
        }
        None => {
            let summaries = manager.verify_all_models().await?;
            
            if summaries.is_empty() {
                println!("No models found to verify.");
                return Ok(());
            }
            
            let has_failures = summaries.iter().any(|s| !s.is_valid);
            for summary in &summaries {
                println!("{}", summary.to_string());
            }
            
            // Exit with error if any model failed verification
            if has_failures {
                std::process::exit(1);
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio_test;

    #[tokio::test]
    async fn test_model_verifier_creation() {
        // ModelVerifier::new requires a config.json in the directory
        let temp_dir = TempDir::new().unwrap();
        let result = ModelVerifier::new(temp_dir.path()).await;
        // Should fail since temp_dir has no config.json
        assert!(result.is_err());
    }

    #[test]
    fn test_checksum_algorithm_parsing() {
        // Test checksum algorithm enum creation
        let sha256 = ChecksumAlgorithm::Sha256;
        let sha1 = ChecksumAlgorithm::Sha1;
        let md5 = ChecksumAlgorithm::Md5;
        
        assert_eq!(sha256, ChecksumAlgorithm::Sha256);
        assert_ne!(sha256, sha1);
        assert_ne!(sha256, md5);
    }

    #[tokio::test]
    async fn test_model_verification_summary() {
        let temp_dir = TempDir::new().unwrap();
        let summary = ModelVerificationSummary {
            model_path: temp_dir.path().to_path_buf(),
            total_files: 5,
            valid_files: 5,
            invalid_files: 0,
            is_valid: true,
        };
        
        let summary_str = summary.to_string();
        assert!(summary_str.contains("VERIFIED"));
        assert!(summary_str.contains("5"));
        assert!(summary_str.contains("5"));
        assert!(summary_str.contains("0"));
        
        let invalid_summary = ModelVerificationSummary {
            model_path: temp_dir.path().to_path_buf(),
            total_files: 5,
            valid_files: 3,
            invalid_files: 2,
            is_valid: false,
        };
        
        let invalid_str = invalid_summary.to_string();
        assert!(invalid_str.contains("VERIFICATION FAILED"));
        assert!(invalid_str.contains("3"));
        assert!(invalid_str.contains("2"));
    }

    #[test]
    fn test_model_integrity_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = ModelIntegrityManager::new(temp_dir.path());
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_checksum_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        
        // Create a test file and config.json so ModelVerifier::new succeeds
        tokio_fs::write(&test_file, "test content").await.unwrap();
        tokio_fs::write(temp_dir.path().join("config.json"), "{}").await.unwrap();
        
        let result = ModelVerifier::new(temp_dir.path()).await;
        assert!(result.is_ok());
        
        let verifier = result.unwrap();
        let checksums = verifier.calculate_checksums().await;
        assert!(checksums.is_ok());
        
        let checksums = checksums.unwrap();
        assert_eq!(checksums.len(), 2);
        assert!(checksums.contains_key("test.txt"));
        
        // Verify the hash is valid hex
        for hash in checksums.values() {
            assert!(hex::decode(hash).is_ok());
        }
    }
}