//! Device selection and management for model inference
//!
//! This module handles device selection (CPU/GPU) with automatic fallback
//! based on feature flags and hardware availability.

use crate::error::{ModelError, Result};
use crate::local::DevicePreference;
use candle_core::Device;
use tracing::{info, warn};

/// Get the appropriate device based on preference and availability.
///
/// # Arguments
/// * `pref` - Device preference (Auto, Cpu, Metal, or Cuda)
/// * `index` - GPU device index (for Metal/Cuda)
///
/// # Returns
/// The selected Candle Device
///
/// # Device Selection Logic
/// - `Auto`: Tries Metal → CUDA → CPU (first available)
/// - `Cpu`: Always returns CPU device
/// - `Metal`: Returns Metal GPU (requires `metal` feature)
/// - `Cuda`: Returns CUDA GPU (requires `cuda` feature)
///
/// # Errors
/// Returns `ModelError` if:
/// - Metal/CUDA requested but feature not enabled
/// - Metal/CUDA requested but device not available
pub fn get_device(pref: DevicePreference, index: usize) -> Result<Device> {
    match pref {
        DevicePreference::Cpu => {
            info!("Using CPU");
            return Ok(Device::Cpu);
        }
        DevicePreference::Metal => {
            #[cfg(feature = "metal")]
            {
                let device = Device::new_metal(index).map_err(|e| {
                    ModelError::LocalModelError(format!("Metal GPU not available: {}", e))
                })?;
                info!("Using Metal GPU");
                return Ok(device);
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(ModelError::InvalidConfig(
                    "Metal support not enabled. Build with --features metal".to_string(),
                ));
            }
        }
        DevicePreference::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = Device::new_cuda(index).map_err(|e| {
                    ModelError::LocalModelError(format!("CUDA GPU not available: {}", e))
                })?;
                info!("Using CUDA GPU");
                return Ok(device);
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(ModelError::InvalidConfig(
                    "CUDA support not enabled. Build with --features cuda".to_string(),
                ));
            }
        }
        DevicePreference::Mlx => {
            #[cfg(feature = "mlx")]
            {
                info!("Using MLX (Apple Silicon unified memory)");
                return Ok(Device::Cpu);
            }
            #[cfg(not(feature = "mlx"))]
            {
                return Err(ModelError::InvalidConfig(
                    "MLX support not enabled. Build with --features mlx".to_string(),
                ));
            }
        }
        DevicePreference::Auto => {}
    }

    // Auto mode: try available GPUs in order, fallback to CPU
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(index) {
            Ok(device) => {
                info!("Auto-selected Metal GPU");
                return Ok(device);
            }
            Err(e) => {
                warn!("Metal GPU not available: {}, trying other options", e);
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(index) {
            Ok(device) => {
                info!("Auto-selected CUDA GPU");
                return Ok(device);
            }
            Err(e) => {
                warn!("CUDA GPU not available: {}, falling back to CPU", e);
            }
        }
    }

    info!("Auto-selected CPU");
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_device_cpu() {
        let result = get_device(DevicePreference::Cpu, 0);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Device::Cpu));
    }

    #[test]
    fn test_get_device_auto() {
        let result = get_device(DevicePreference::Auto, 0);
        assert!(result.is_ok());
        // Will succeed with either GPU or CPU depending on availability
    }

    #[test]
    fn test_device_preference_from_str() {
        use std::str::FromStr;

        assert!(matches!(
            DevicePreference::from_str("auto"),
            Ok(DevicePreference::Auto)
        ));
        assert!(matches!(
            DevicePreference::from_str("cpu"),
            Ok(DevicePreference::Cpu)
        ));
        assert!(matches!(
            DevicePreference::from_str("metal"),
            Ok(DevicePreference::Metal)
        ));
        assert!(matches!(
            DevicePreference::from_str("cuda"),
            Ok(DevicePreference::Cuda)
        ));
        assert!(matches!(
            DevicePreference::from_str("mlx"),
            Ok(DevicePreference::Mlx)
        ));
        assert!(matches!(
            DevicePreference::from_str("AUTO"),
            Ok(DevicePreference::Auto)
        ));
        assert!(matches!(
            DevicePreference::from_str("CPU"),
            Ok(DevicePreference::Cpu)
        ));

        assert!(DevicePreference::from_str("invalid").is_err());
    }
}
