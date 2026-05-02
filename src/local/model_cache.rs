//! In-memory model caching
//!
//! This module provides a global model cache to avoid reloading models
//! between requests, significantly improving performance for repeated inference.

use crate::error::Result;
use crate::local::{LocalModel, LocalModelConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use tokio::sync::RwLock;

/// Global model cache instance
static MODEL_CACHE: OnceLock<ModelCache> = OnceLock::new();

/// Cached model entry with metadata
struct CachedModel {
    /// The loaded model
    model: Arc<RwLock<LocalModel>>,
    /// When this model was last accessed
    last_accessed: Instant,
    /// When this model was loaded
    loaded_at: Instant,
    /// Number of times this model has been accessed
    access_count: usize,
}

/// Global in-memory model cache
///
/// This cache maintains loaded models in memory to avoid the overhead
/// of reloading them between requests. Models are automatically evicted
/// when the cache is full or when they haven't been used recently.
pub struct ModelCache {
    /// Map from model path to cached model
    cache: Mutex<HashMap<PathBuf, CachedModel>>,
    /// Maximum number of models to keep in cache
    max_cached_models: usize,
    /// Maximum age before a model is evicted (unused)
    max_idle_duration: Duration,
    /// Whether caching is enabled
    enabled: Mutex<bool>,
}

impl ModelCache {
    /// Create a new model cache with default settings
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_cached_models: 3, // Default: cache up to 3 models
            max_idle_duration: Duration::from_secs(3600), // 1 hour
            enabled: Mutex::new(true),
        }
    }

    /// Configure the cache with custom settings
    pub fn with_config(max_cached_models: usize, max_idle_duration: Duration) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_cached_models,
            max_idle_duration,
            enabled: Mutex::new(true),
        }
    }

    /// Enable or disable model caching
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.lock().unwrap() = enabled;
        info!("Model caching {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().unwrap()
    }

    /// Get a model from cache, loading it if necessary
    ///
    /// This is the main entry point for cached model access.
    /// Returns a clone of the Arc wrapping the model for thread-safe access.
    pub async fn get_or_load(&self, config: LocalModelConfig) -> Result<Arc<RwLock<LocalModel>>> {
        if !self.is_enabled() {
            debug!("Caching disabled, loading model directly");
            let model = LocalModel::load(config).await?;
            return Ok(Arc::new(RwLock::new(model)));
        }

        let model_path = config.model_path.clone();

        // Check if model is already cached
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get_mut(&model_path) {
                cached.last_accessed = Instant::now();
                cached.access_count += 1;
                debug!(
                    "Cache hit for model '{}' (access #{})",
                    model_path.display(),
                    cached.access_count
                );
                return Ok(cached.model.clone());
            }
        }

        // Cache miss - load the model
        debug!("Cache miss for model '{}', loading...", model_path.display());
        let model = LocalModel::load(config).await?;
        let model_arc = Arc::new(RwLock::new(model));

        // Add to cache
        {
            let mut cache = self.cache.lock().unwrap();
            let cached = CachedModel {
                model: model_arc.clone(),
                last_accessed: Instant::now(),
                loaded_at: Instant::now(),
                access_count: 1,
            };

            // Evict old models if necessary
            self.evict_if_needed(&mut cache);

            cache.insert(model_path, cached);
            info!(
                "Loaded and cached model ({} models in cache)",
                cache.len()
            );
        }

        Ok(model_arc)
    }

    /// Get a model from the cache without loading
    ///
    /// Returns None if the model is not cached.
    pub fn get_cached(&self, model_path: &PathBuf) -> Option<Arc<RwLock<LocalModel>>> {
        if !self.is_enabled() {
            return None;
        }

        let mut cache = self.cache.lock().unwrap();
        if let Some(cached) = cache.get_mut(model_path) {
            cached.last_accessed = Instant::now();
            cached.access_count += 1;
            debug!(
                "Cache hit for model '{}' (access #{})",
                model_path.display(),
                cached.access_count
            );
            return Some(cached.model.clone());
        }
        None
    }

    /// Explicitly preload a model into the cache
    ///
    /// Useful for warming up the cache with commonly used models.
    pub async fn preload(&self, config: LocalModelConfig) -> Result<Arc<RwLock<LocalModel>>> {
        info!("Preloading model '{}'", config.model_path.display());
        self.get_or_load(config).await
    }

    /// Remove a specific model from the cache
    pub fn evict(&self, model_path: &PathBuf) {
        let mut cache = self.cache.lock().unwrap();
        if cache.remove(model_path).is_some() {
            info!("Evicted model '{}' from cache", model_path.display());
        }
    }

    /// Clear all cached models
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        let count = cache.len();
        cache.clear();
        info!("Cleared all {} cached model(s)", count);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let now = Instant::now();

        let models: Vec<_> = cache.iter().map(|(path, cached)| {
            CacheModelInfo {
                path: path.clone(),
                access_count: cached.access_count,
                last_accessed: now.duration_since(cached.last_accessed),
                loaded_at: now.duration_since(cached.loaded_at),
            }
        }).collect();

        CacheStats {
            cached_models: cache.len(),
            max_cached_models: self.max_cached_models,
            enabled: self.is_enabled(),
            models,
        }
    }

    /// Clean up models that haven't been accessed recently
    pub fn cleanup_idle(&self) {
        let mut cache = self.cache.lock().unwrap();
        let now = Instant::now();

        let idle_models: Vec<PathBuf> = cache
            .iter()
            .filter(|(_, cached)| {
                now.duration_since(cached.last_accessed) > self.max_idle_duration
            })
            .map(|(path, _)| path.clone())
            .collect();

        for path in idle_models {
            cache.remove(&path);
            info!("Removed idle model '{}' from cache", path.display());
        }
    }

    /// Evict models if the cache is at capacity
    fn evict_if_needed(&self, cache: &mut HashMap<PathBuf, CachedModel>) {
        if cache.len() >= self.max_cached_models {
            // Find the least recently used model
            if let Some((lru_path, _)) = cache
                .iter()
                .min_by_key(|(_, cached)| cached.last_accessed)
            {
                let path = lru_path.clone();
                cache.remove(&path);
                warn!(
                    "Evicted LRU model '{}' from cache (capacity: {})",
                    path.display(),
                    self.max_cached_models
                );
            }
        }
    }
}

/// Statistics about the model cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of models currently cached
    pub cached_models: usize,
    /// Maximum number of models that can be cached
    pub max_cached_models: usize,
    /// Whether caching is enabled
    pub enabled: bool,
    /// Information about each cached model
    pub models: Vec<CacheModelInfo>,
}

/// Information about a single cached model
#[derive(Debug, Clone)]
pub struct CacheModelInfo {
    /// Path to the model
    pub path: PathBuf,
    /// Number of times the model has been accessed
    pub access_count: usize,
    /// Time since last access
    pub last_accessed: Duration,
    /// Time since the model was loaded
    pub loaded_at: Duration,
}

/// Get the global model cache instance
pub fn global_model_cache() -> &'static ModelCache {
    MODEL_CACHE.get_or_init(|| ModelCache::new())
}

/// Convenience function to get or load a model using the global cache
pub async fn get_or_load_model(config: LocalModelConfig) -> Result<Arc<RwLock<LocalModel>>> {
    global_model_cache().get_or_load(config).await
}

/// Convenience function to get a cached model without loading
pub fn get_cached_model(model_path: &PathBuf) -> Option<Arc<RwLock<LocalModel>>> {
    global_model_cache().get_cached(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = ModelCache::new();
        assert!(cache.is_enabled());
        assert_eq!(cache.max_cached_models, 3);
    }

    #[test]
    fn test_cache_configuration() {
        let cache = ModelCache::with_config(5, Duration::from_secs(7200));
        assert_eq!(cache.max_cached_models, 5);
        assert_eq!(cache.max_idle_duration.as_secs(), 7200);
    }

    #[test]
    fn test_enable_disable() {
        let cache = ModelCache::new();
        assert!(cache.is_enabled());

        cache.set_enabled(false);
        assert!(!cache.is_enabled());

        cache.set_enabled(true);
        assert!(cache.is_enabled());
    }

    #[test]
    fn test_evict() {
        let cache = ModelCache::new();
        let path = PathBuf::from("/test/model");

        // Evicting non-existent model should not error
        cache.evict(&path);

        // Stats should show 0 models
        let stats = cache.stats();
        assert_eq!(stats.cached_models, 0);
    }

    #[test]
    fn test_clear() {
        let cache = ModelCache::new();
        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.cached_models, 0);
        assert!(stats.enabled);
    }

    #[test]
    fn test_stats() {
        let cache = ModelCache::new();
        let stats = cache.stats();

        assert_eq!(stats.cached_models, 0);
        assert_eq!(stats.max_cached_models, 3);
        assert!(stats.enabled);
        assert_eq!(stats.models.len(), 0);
    }

    #[test]
    fn test_cleanup_idle() {
        let cache = ModelCache::with_config(10, Duration::from_secs(1));
        cache.cleanup_idle();

        // Should not error even with empty cache
        let stats = cache.stats();
        assert_eq!(stats.cached_models, 0);
    }
}
