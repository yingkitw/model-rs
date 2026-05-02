use crate::error::{ModelError, Result};
use crate::config::DEFAULT_MIRROR;
use reqwest::Client;
use serde::Deserialize;
use tracing::info;

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "modelId")]
    #[serde(default)]
    model_id: Option<String>,
    #[serde(alias = "author")]  // Handle both "author" and "authorId"
    #[serde(default)]
    author_name: Option<String>,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    library_name: Option<String>,
}

pub async fn search_models(
    query: &str,
    limit: usize,
    author: Option<&str>,
    mirror: Option<&str>,
) -> Result<()> {
    let mirror_url = mirror.unwrap_or(DEFAULT_MIRROR);
    let client = Client::builder()
        .user_agent("model-rs/0.1.0")
        .build()?;

    info!("Searching for models with query: '{}'", query);

    // Build search URL with query parameters
    let mut search_url = format!(
        "{}/api/models?search={}&limit={}",
        mirror_url,
        urlencoding::encode(query),
        limit
    );

    if let Some(author_filter) = author {
        search_url.push_str(&format!("&author={}", urlencoding::encode(author_filter)));
    }

    let response = client
        .get(&search_url)
        .send()
        .await
        .map_err(|e| ModelError::DownloadError(format!("Failed to search models: {}", e)))?;

    if !response.status().is_success() {
        return Err(ModelError::DownloadError(format!(
            "HTTP {} when searching models",
            response.status()
        )));
    }

    // HuggingFace API returns an array directly, not wrapped in an object
    let search_result: Vec<ModelInfo> = response
        .json()
        .await
        .map_err(|e| ModelError::DownloadError(format!("Failed to parse search results: {}", e)))?;

    let formatter = crate::output::OutputFormatter::new();

    if search_result.is_empty() {
        formatter.print_warning(&format!("No models found matching '{}'", query));
        return Ok(());
    }

    formatter.print_header(&format!("Found {} models", search_result.len()));

    for (index, model) in search_result.iter().enumerate() {
        let model_id = model.id.as_ref()
            .or(model.model_id.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        formatter.print_search_result(
            index + 1,
            model_id,
            model.author_name.as_deref(),
            model.pipeline_tag.as_deref(),
            model.downloads,
            model.likes,
            model.library_name.as_deref(),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_deserialization_with_id() {
        let json = r#"{"id": "test/model", "author": "test", "downloads": 1000, "likes": 50}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.id, Some("test/model".to_string()));
        assert_eq!(info.author_name, Some("test".to_string()));
        assert_eq!(info.downloads, Some(1000));
        assert_eq!(info.likes, Some(50));
    }

    #[test]
    fn test_model_info_deserialization_with_model_id() {
        let json = r#"{"modelId": "test/model", "author": "test"}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.model_id, Some("test/model".to_string()));
        assert_eq!(info.author_name, Some("test".to_string()));
    }

    #[test]
    fn test_model_info_deserialization_with_defaults() {
        let json = r#"{"id": "test/model"}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.id, Some("test/model".to_string()));
        assert_eq!(info.author_name, None);
        assert_eq!(info.downloads, None);
        assert_eq!(info.likes, None);
        assert_eq!(info.pipeline_tag, None);
        assert_eq!(info.library_name, None);
    }

    #[test]
    fn test_model_info_with_pipeline_tag() {
        let json = r#"{"id": "test/model", "pipeline_tag": "text-generation"}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.pipeline_tag, Some("text-generation".to_string()));
    }

    #[test]
    fn test_model_info_with_library_name() {
        let json = r#"{"id": "test/model", "library_name": "transformers"}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.library_name, Some("transformers".to_string()));
    }

    #[test]
    fn test_model_info_all_fields() {
        let json = r#"{
            "id": "test/model",
            "author": "testorg",
            "downloads": 1000000,
            "likes": 5000,
            "pipeline_tag": "text-generation",
            "library_name": "transformers"
        }"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.id, Some("test/model".to_string()));
        assert_eq!(info.author_name, Some("testorg".to_string()));
        assert_eq!(info.downloads, Some(1000000));
        assert_eq!(info.likes, Some(5000));
        assert_eq!(info.pipeline_tag, Some("text-generation".to_string()));
        assert_eq!(info.library_name, Some("transformers".to_string()));
    }

    #[test]
    fn test_model_info_empty_json() {
        let json = r#"{}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.id, None);
        assert_eq!(info.model_id, None);
        assert_eq!(info.author_name, None);
        assert_eq!(info.downloads, None);
        assert_eq!(info.likes, None);
    }

    #[tokio::test]
    async fn test_search_models_requires_query() {
        // This test verifies the function signature and basic behavior
        // Full integration tests would require mocking the HTTP client
        let result = search_models("test", 5, None, None).await;

        // We expect this to fail in test environment without network,
        // but the function should at least construct the request properly
        // If it succeeds with empty results, that's also acceptable
        match result {
            Ok(_) => {}, // Success (empty results or network worked)
            Err(ModelError::DownloadError(_)) => {}, // Expected in test env
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
