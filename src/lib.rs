//! model-rs library crate.
//!
//! The CLI binary (`main.rs`) delegates to [`run`]. Exposing a library target enables
//! `cargo bench` (Criterion) and future programmatic use without duplicating modules.

pub mod cli;
pub mod config;
pub mod config_file;
pub mod download;
pub mod error;
pub mod format;
pub mod influencer;
pub mod local;
pub mod model_ops;
pub mod models;
pub mod output;
pub mod search;
pub mod validation;
pub mod verification;
pub mod version_manager;

pub use error::Result;

use clap::Parser;
use cli::{Cli, Commands, VersionCommands, ConfigCommands};
use dotenvy::dotenv;
use std::path::PathBuf;
use tracing::warn;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use validation::*;
use version_manager::*;
use config_file::*;
use crate::error::ModelError;

/// Resolve the model path for a `chat`/`run` invocation:
/// - If a HF id is supplied, validate it and resolve to a cache path.
/// - Otherwise fall back to `MODEL_RS_MODEL_PATH`.
async fn resolve_chat_model_path(model: Option<String>) -> Result<PathBuf> {
    let ops = model_ops::ModelOperations::new();
    match model {
        Some(m) => {
            validate_model_name(&m)?;
            ops.resolve_model_path(&m)
        }
        None => config::get_model_path().ok_or_else(|| {
            ModelError::invalid_config(
                "Model path is required. Pass a model to `run` or set MODEL_RS_MODEL_PATH.",
            )
        }),
    }
}

/// Validate the standard generation-parameter bundle used by `chat`/`run`.
fn validate_chat_generation(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    device: &str,
    device_index: usize,
) -> Result<()> {
    validate_generation_args(max_tokens, temperature, top_p, top_k, repeat_penalty, device, device_index)
}

/// Run the full CLI: load `.env`, parse arguments, initialize tracing, dispatch subcommands.
pub async fn run() -> Result<()> {
    let _ = dotenv();

    let cli = Cli::parse();

    let log_level = match &cli.command {
        Commands::Generate { .. } | Commands::Chat { .. } | Commands::Run { .. } => {
            "model_rs=warn"
        }
        _ => "model_rs=info",
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::Download { model, mirror, output } => {
            // Validate model name
            validate_model_name(&model)?;

            // Validate output directory if provided
            if let Some(ref path) = output {
                validate_path(path, "output")?;

                // Create directory if it doesn't exist
                if !tokio::fs::try_exists(path).await? {
                    tokio::fs::create_dir_all(path).await.map_err(|e| {
                        ModelError::validation_error(
                            path.to_string_lossy().as_ref(),
                            &format!("Failed to create output directory: {}", e),
                            "Check file permissions and disk space"
                        )
                    })?;
                }
            }

            let mirror_url = mirror.or_else(|| Some(config::get_mirror()));
            let output_dir = output.or_else(config::get_output_dir);
            download::download_model(&model, mirror_url.as_deref(), output_dir.as_deref()).await?;
        }
        Commands::Search {
            query,
            limit,
            author,
        } => {
            if query.trim().is_empty() {
                return Err(ModelError::validation_error(
                    "query",
                    "Search query cannot be empty",
                    "Provide a non-empty search term",
                ));
            }
            if limit == 0 || limit > 100 {
                return Err(ModelError::validation_error(
                    &limit.to_string(),
                    "Invalid limit value",
                    "Use a value between 1 and 100",
                ));
            }
            if let Some(ref author) = author {
                if author.trim().is_empty() {
                    return Err(ModelError::validation_error(
                        author,
                        "Author name cannot be empty",
                        "Provide a valid author/organization name",
                    ));
                }
            }
            search::search_models(&query, limit, author.as_deref(), None).await?;
        }
        Commands::Serve {
            model_path,
            port,
            device,
            device_index,
        } => {
            validate_port(port)?;
            validate_device(&device)?;
            validate_device_index(Some(device_index as u32))?;
            if let Some(ref path) = model_path {
                validate_model_directory(path).await?;
            }
            let model = model_path.or_else(config::get_model_path);
            influencer::serve(model.as_deref(), port, &device, device_index).await?;
        }
        Commands::Generate {
            prompt,
            system,
            model_path,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            device,
            device_index,
        } => {
            if prompt.trim().is_empty() {
                return Err(ModelError::validation_error(
                    "prompt",
                    "Prompt cannot be empty",
                    "Provide a non-empty prompt for text generation",
                ));
            }
            validate_generation_args(max_tokens, temperature, top_p, top_k, repeat_penalty, &device, device_index)?;
            if let Some(ref path) = model_path {
                validate_model_directory(path).await?;
            }
            let model = model_path.or_else(config::get_model_path);
            influencer::generate(
                &prompt,
                system.as_deref(),
                model.as_deref(),
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &device,
                device_index,
            )
            .await?;
        }
        Commands::Run {
            model,
            system,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            device,
            device_index,
            session,
            save_on_exit,
        } => {
            validate_chat_generation(max_tokens, temperature, top_p, top_k, repeat_penalty, &device, device_index)?;
            let model_path = resolve_chat_model_path(model).await?;
            validate_model_directory(&model_path).await?;
            influencer::chat(
                &model_path,
                system.as_deref(),
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &device,
                device_index,
                session.as_deref(),
                save_on_exit.as_deref(),
            )
            .await?;
        }
        Commands::Stop { model, port, force } => {
            let ops = model_ops::ModelOperations::new();
            ops.stop(model.as_deref(), port, force)?;
        }
        Commands::Chat {
            model_path,
            system,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            device,
            device_index,
            session,
            save_on_exit,
        } => {
            validate_chat_generation(max_tokens, temperature, top_p, top_k, repeat_penalty, &device, device_index)?;
            validate_model_directory(&model_path).await?;
            influencer::chat(
                &model_path,
                system.as_deref(),
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &device,
                device_index,
                session.as_deref(),
                save_on_exit.as_deref(),
            )
            .await?;
        }
        Commands::Embed {
            text,
            model_path,
            device,
            device_index,
        } => {
            // Validate text for embedding
            if text.trim().is_empty() {
                return Err(ModelError::validation_error(
                    "text",
                    "Text cannot be empty",
                    "Provide non-empty text for embedding generation"
                ));
            }
            
            // Validate device and device index
            validate_device(&device)?;
            validate_device_index(Some(device_index as u32))?;
            
            // Validate model directory
            validate_model_directory(&model_path).await?;
            
            influencer::embed(&text, &model_path, &device, device_index).await?;
        }
        Commands::List { models_dir } => {
            let models_dir_path = models_dir.as_deref();
            let models = models::list_models(models_dir_path)?;
            let formatter = output::OutputFormatter::new();

            if models.is_empty() {
                formatter.print_warning("No models found.");
                formatter.print_markdown("\n**To download a model:**\n\n```bash\nmodel-rs download <model-name>\n```\n\n**Example:**\n\n```bash\nmodel-rs download TinyLlama/TinyLlama-1.1B-Chat-v1.0\n```\n");
            } else {
                models::display_models(&models, &formatter);
            }
        }
        Commands::Deploy {
            model_path,
            port,
            device,
            device_index,
            detached,
        } => {
            let model = model_path.or_else(config::get_model_path);
            let formatter = output::OutputFormatter::new();

            if detached {
                formatter.print_header("Deploying Model (Background Mode)");
                formatter.print_info(&format!(
                    "Server will be accessible at: http://localhost:{}",
                    port
                ));
                formatter.print_markdown("\n**To stop the server later:**\n\n```bash\nps aux | grep model-rs\nkill <pid>\n```\n");
            }

            influencer::serve(model.as_deref(), port, &device, device_index).await?;

            if detached {
                formatter.print_success("Model deployed successfully!");
                formatter.print_markdown(&format!(
                    "\n**Test the deployment:**\n\n```bash\ncurl http://localhost:{}/health\n```\n",
                    port
                ));
            }
        }
        Commands::Show { model } => {
            // Validate model name
            validate_model_name(&model)?;
            let ops = model_ops::ModelOperations::new();
            ops.show(&model)?;
        }
        Commands::Remove { model, force } => {
            // Validate model name
            validate_model_name(&model)?;
            let ops = model_ops::ModelOperations::new();
            ops.remove(&model, force)?;
        }
        Commands::Ps => {
            let ops = model_ops::ModelOperations::new();
            ops.ps()?;
        }
        Commands::Copy {
            source,
            destination,
        } => {
            // Validate both source and destination model names
            validate_model_name(&source)?;
            validate_model_name(&destination)?;
            
            let ops = model_ops::ModelOperations::new();
            ops.copy(&source, &destination)?;
        }
        Commands::Info { model } => {
            // Validate model name
            validate_model_name(&model)?;
            let ops = model_ops::ModelOperations::new();
            ops.info(&model)?;
        }
        Commands::Verify { model } => {
            // Validate model name
            validate_model_name(&model)?;
            
            let models_dir_path = config_file::get_models_dir();
            let manager = verification::ModelIntegrityManager::new(&models_dir_path)?;
            
            let summary = manager.verify_model(&model).await?;
            println!("{}", summary.to_string());
            
            if !summary.is_valid {
                std::process::exit(1);
            }
        }
        Commands::GenerateChecksums { model } => {
            // Validate model name
            validate_model_name(&model)?;
            
            let models_dir_path = config_file::get_models_dir();
            let manager = verification::ModelIntegrityManager::new(&models_dir_path)?;
            
            manager.generate_checksums(&model).await?;
            println!("Checksums generated for model: {}", model);
            
            // Also verify the model after generating checksums
            let summary = manager.verify_model(&model).await?;
            println!("Verification after checksum generation:");
            println!("{}", summary.to_string());
            
            if !summary.is_valid {
                warn!("Some files failed verification after checksum generation");
            }
        }
        Commands::Cache {
            stats,
            clear,
            enable,
            preload,
            evict,
            max,
        } => {
            let formatter = output::OutputFormatter::new();
            formatter.print_header("Model Cache");

            if let Some(enabled) = enable {
                local::global_model_cache().set_enabled(enabled);
                formatter.print_success(&format!(
                    "Caching {}",
                    if enabled { "enabled" } else { "disabled" }
                ));
            }

            if clear {
                local::global_model_cache().clear();
                formatter.print_success("Cache cleared");
            }

            if stats
                || (!clear
                    && enable.is_none()
                    && preload.is_none()
                    && evict.is_none()
                    && max.is_none())
            {
                let cache_stats = local::global_model_cache().stats();
                println!("\n### Cache Status");
                println!(
                    "- **Status:** {}",
                    if cache_stats.enabled {
                        "Enabled"
                    } else {
                        "Disabled"
                    }
                );
                println!(
                    "- **Cached Models:** {} / {}",
                    cache_stats.cached_models, cache_stats.max_cached_models
                );

                if !cache_stats.models.is_empty() {
                    println!("\n### Cached Models");
                    for model_info in &cache_stats.models {
                        println!("\n#### {}", model_info.path.display());
                        println!("- **Access Count:** {}", model_info.access_count);
                        println!(
                            "- **Last Accessed:** {}s ago",
                            model_info.last_accessed.as_secs()
                        );
                        println!("- **Loaded:** {}s ago", model_info.loaded_at.as_secs());
                    }
                } else {
                    println!("\nNo models cached.");
                }
            }

            if let Some(model_name) = preload {
                formatter.print_info(&format!("Preloading model '{}'...", model_name));
                let ops = model_ops::ModelOperations::new();
                let model_path = ops.resolve_model_path(&model_name)?;
                let config = local::LocalModelConfig {
                    model_path,
                    ..Default::default()
                };
                let _model = local::global_model_cache().preload(config).await?;
                formatter.print_success(&format!(
                    "Model '{}' preloaded into cache",
                    model_name
                ));
            }

            if let Some(model_name) = evict {
                let ops = model_ops::ModelOperations::new();
                let model_path = ops.resolve_model_path(&model_name)?;
                local::global_model_cache().evict(&model_path);
                formatter.print_success(&format!("Model '{}' evicted from cache", model_name));
            }

            if let Some(max_models) = max {
                local::global_model_cache().set_max_cached_models(max_models);
                formatter.print_success(&format!(
                    "Max cached models set to {} (takes effect on next insert)",
                    max_models
                ));
            }
        }
        Commands::Versions { command } => {
            let models_dir_path = config_file::get_models_dir();
            let version_cli = VersionManagerCLI::new(&models_dir_path)?;
            
            match command {
                VersionCommands::List { model } => {
                    version_cli.list_versions(model.as_deref()).await?;
                }
                VersionCommands::Pin { model, version } => {
                    version_cli.pin_version(&model, &version).await?;
                }
                VersionCommands::Unpin { model, version } => {
                    version_cli.unpin_version(&model, &version).await?;
                }
                VersionCommands::Stats => {
                    version_cli.show_statistics().await?;
                }
                VersionCommands::Cleanup { model, keep_latest, keep_pinned } => {
                    version_cli.cleanup_versions(model.as_deref(), keep_latest, keep_pinned).await?;
                }
            }
        }
        Commands::Config { command } => {
            match command {
                ConfigCommands::Init => {
                    let mut manager = ConfigManager::new()?;
                    let config_file = manager.create_config_file().await?;
                    println!("✅ Configuration file created at: {}", config_file.display());
                    println!("📝 Edit the file with: model-rs config edit");
                }
                ConfigCommands::Edit => {
                    let mut manager = ConfigManager::new()?;
                    manager.edit_config_file().await?;
                    println!("✅ Configuration updated");
                }
                ConfigCommands::Show => {
                    let mut manager = ConfigManager::new()?;
                    let config = manager.get_merged_config()?;
                    let toml_string = toml::to_string_pretty(&config)
                        .map_err(|e| ModelError::invalid_config(
                            format!("Failed to serialize config: {}", e),
                        ))?;
                    println!("{}", toml_string);
                }
                ConfigCommands::Sources => {
                    let manager = ConfigManager::new()?;
                    let sources = manager.get_config_sources();
                    
                    println!("Configuration sources:");
                    for (key, source) in sources {
                        println!("  {}: {:?}", key, source);
                    }
                }
                ConfigCommands::Validate => {
                    let mut manager = ConfigManager::new()?;
                    let config = manager.get_merged_config()?;

                    match manager.validate_configuration(&config) {
                        Ok(_) => {
                            println!("✅ Configuration is valid");
                        }
                        Err(e) => {
                            println!("❌ Configuration validation failed:");
                            println!("   {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                ConfigCommands::Env => {
                    let formatter = output::OutputFormatter::new();
                    formatter.print_header("Environment-Derived Configuration (MODEL_RS_*)");
                    let md = config::env_config_markdown();
                    for chunk in md.split("\n\n") {
                        formatter.print_markdown(chunk);
                    }
                }
                ConfigCommands::Reset => {
                    let mut manager = ConfigManager::new()?;
                    let default_config = AppConfig::default();
                    manager.save_configuration(&default_config).await?;
                    println!("✅ Configuration reset to defaults");
                }
            }
        }
    }

    Ok(())
}
