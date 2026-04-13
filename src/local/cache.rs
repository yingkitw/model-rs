//! Session-based KV cache management
//!
//! This module provides efficient cache reuse for multi-turn conversations,
//! addressing the limitation of creating fresh caches per request.

use crate::error::{InfluenceError, Result};
use crate::local::config::ModelArchitecture;
use crate::local::backends::LocalBackend;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

use candle_core::Device;

/// Cache handle for managing session-based cache reuse
#[derive(Clone)]
pub struct CacheHandle {
    session_id: String,
    last_used: Arc<Mutex<Instant>>,
}

impl CacheHandle {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            last_used: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn update_last_used(&self) {
        *self.last_used.lock().unwrap() = Instant::now();
    }

    pub fn last_used(&self) -> Instant {
        *self.last_used.lock().unwrap()
    }
}

/// Cache engine for managing session-based KV caches
pub struct CacheEngine {
    caches: Arc<Mutex<HashMap<String, CachedSession>>>,
    max_cache_age: Duration,
    max_sessions: usize,
    device: Device,
}

/// A cached session with its associated cache data
struct CachedSession {
    handle: CacheHandle,
    token_ids: Vec<u32>,
    created_at: Instant,
}

impl CacheEngine {
    /// Create a new cache engine with default settings
    pub fn new(device: Device) -> Self {
        Self::with_config(device, Duration::from_secs(3600), 100)
    }

    /// Create a new cache engine with custom configuration
    pub fn with_config(device: Device, max_cache_age: Duration, max_sessions: usize) -> Self {
        Self {
            caches: Arc::new(Mutex::new(HashMap::new())),
            max_cache_age,
            max_sessions,
            device,
        }
    }

    /// Get or create a cache handle for the given session
    pub fn get_cache(&self, session_id: &str) -> CacheHandle {
        let mut caches = self.caches.lock().unwrap();
        let now = Instant::now();

        // Clean up expired sessions
        self.cleanup_expired(&mut caches, now);

        // Get or create session
        if let Some(session) = caches.get(session_id) {
            session.handle.update_last_used();
            return session.handle.clone();
        }

        // Enforce max sessions limit (evict oldest if needed)
        if caches.len() >= self.max_sessions {
            self.evict_oldest(&mut caches);
        }

        let handle = CacheHandle::new(session_id.to_string());
        caches.insert(session_id.to_string(), CachedSession {
            handle: handle.clone(),
            token_ids: Vec::new(),
            created_at: now,
        });

        debug!("Created new cache session: {}", session_id);
        handle
    }

    /// Update the token IDs for a session
    pub fn update_tokens(&self, session_id: &str, token_ids: Vec<u32>) {
        let mut caches = self.caches.lock().unwrap();
        if let Some(session) = caches.get_mut(session_id) {
            session.token_ids = token_ids;
            session.handle.update_last_used();
        }
    }

    /// Get the cached token IDs for a session
    pub fn get_tokens(&self, session_id: &str) -> Option<Vec<u32>> {
        let caches = self.caches.lock().unwrap();
        caches.get(session_id).map(|s| s.token_ids.clone())
    }

    /// Invalidate a specific session cache
    pub fn invalidate(&self, session_id: &str) {
        let mut caches = self.caches.lock().unwrap();
        if caches.remove(session_id).is_some() {
            debug!("Invalidated cache session: {}", session_id);
        }
    }

    /// Clear all session caches
    pub fn clear_all(&self) {
        let mut caches = self.caches.lock().unwrap();
        let count = caches.len();
        caches.clear();
        debug!("Cleared all {} cache sessions", count);
    }

    /// Get statistics about the cache engine
    pub fn stats(&self) -> CacheStats {
        let caches = self.caches.lock().unwrap();
        let now = Instant::now();
        CacheStats {
            total_sessions: caches.len(),
            active_sessions: caches.values().filter(|s| now.duration_since(s.handle.last_used()) < Duration::from_secs(300)).count(),
            oldest_session_age: caches.values().map(|s| now.duration_since(s.created_at)).min().unwrap_or_default(),
        }
    }

    /// Clean up expired sessions
    fn cleanup_expired(&self, caches: &mut HashMap<String, CachedSession>, now: Instant) {
        let expired: Vec<String> = caches
            .iter()
            .filter(|(_, s)| now.duration_since(s.handle.last_used()) > self.max_cache_age)
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            caches.remove(&id);
            debug!("Removed expired cache session: {}", id);
        }
    }

    /// Evict the oldest session
    fn evict_oldest(&self, caches: &mut HashMap<String, CachedSession>) {
        if let Some((oldest_id, _)) = caches
            .iter()
            .min_by_key(|(_, s)| s.created_at)
            .map(|(id, _)| (id.clone(), ()))
        {
            caches.remove(&oldest_id);
            warn!("Evicted oldest cache session due to size limit: {}", oldest_id);
        }
    }
}

/// Statistics about the cache engine
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub oldest_session_age: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_handle_creation() {
        let handle = CacheHandle::new("test-session".to_string());
        assert_eq!(handle.session_id(), "test-session");
    }

    #[test]
    fn test_cache_handle_update_last_used() {
        let handle = CacheHandle::new("test-session".to_string());
        let initial = handle.last_used();
        std::thread::sleep(Duration::from_millis(10));
        handle.update_last_used();
        assert!(handle.last_used() > initial);
    }

    #[test]
    fn test_cache_engine_creation() {
        let device = Device::Cpu;
        let engine = CacheEngine::new(device.clone());
        assert_eq!(engine.max_sessions, 100);

        let engine_custom = CacheEngine::with_config(device, Duration::from_secs(7200), 50);
        assert_eq!(engine_custom.max_sessions, 50);
    }

    #[test]
    fn test_cache_get_or_create() {
        let device = Device::Cpu;
        let engine = CacheEngine::new(device);

        let handle1 = engine.get_cache("session-1");
        let handle2 = engine.get_cache("session-1");
        let handle3 = engine.get_cache("session-2");

        assert_eq!(handle1.session_id(), "session-1");
        assert_eq!(handle2.session_id(), "session-1");
        assert_eq!(handle3.session_id(), "session-2");
    }

    #[test]
    fn test_cache_update_and_get_tokens() {
        let device = Device::Cpu;
        let engine = CacheEngine::new(device);

        engine.get_cache("session-1");
        let tokens = vec![1, 2, 3, 4, 5];
        engine.update_tokens("session-1", tokens.clone());

        let retrieved = engine.get_tokens("session-1");
        assert_eq!(retrieved, Some(tokens));
    }

    #[test]
    fn test_cache_invalidate() {
        let device = Device::Cpu;
        let engine = CacheEngine::new(device);

        engine.get_cache("session-1");
        engine.invalidate("session-1");

        assert_eq!(engine.get_tokens("session-1"), None);
    }

    #[test]
    fn test_cache_clear_all() {
        let device = Device::Cpu;
        let engine = CacheEngine::new(device);

        engine.get_cache("session-1");
        engine.get_cache("session-2");
        engine.clear_all();

        let stats = engine.stats();
        assert_eq!(stats.total_sessions, 0);
    }

    #[test]
    fn test_cache_stats() {
        let device = Device::Cpu;
        let engine = CacheEngine::new(device);

        engine.get_cache("session-1");
        engine.get_cache("session-2");

        let stats = engine.stats();
        assert_eq!(stats.total_sessions, 2);
    }

    #[test]
    fn test_max_sessions_limit() {
        let device = Device::Cpu;
        let engine = CacheEngine::with_config(device, Duration::from_secs(3600), 3);

        engine.get_cache("session-1");
        engine.get_cache("session-2");
        engine.get_cache("session-3");
        engine.get_cache("session-4"); // Should evict oldest

        let stats = engine.stats();
        assert_eq!(stats.total_sessions, 3);
    }
}
