//! model-rs library crate.
//!
//! The CLI binary (`main.rs`) delegates to [`run`]. Exposing a library target enables
//! `cargo bench` (Criterion) and future programmatic use without duplicating modules.

pub mod cli;
pub mod config;
pub mod download;
pub mod error;
pub mod format;
pub mod influencer;
pub mod local;
pub mod model_ops;
pub mod models;
pub mod output;
pub mod search;

pub use error::Result;

use clap::Parser;
use cli::{Cli, Commands};
use dotenvy::dotenv;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
            let mirror_url = mirror.or_else(|| Some(config::get_mirror()));
            let output_dir = output.or_else(config::get_output_dir);
            download::download_model(&model, mirror_url.as_deref(), output_dir.as_deref()).await?;
        }
        Commands::Search { query, limit, author } => {
            search::search_models(&query, limit, author.as_deref(), None).await?;
        }
        Commands::Serve {
            model_path,
            port,
            device,
            device_index,
        } => {
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
            let ops = model_ops::ModelOperations::new();
            let model_path = match model {
                Some(m) => ops.resolve_model_path(&m)?,
                None => config::get_model_path().ok_or_else(|| {
                    error::ModelError::InvalidConfig(
                        "Model path is required. Pass a model to `run` or set MODEL_RS_MODEL_PATH."
                            .to_string(),
                    )
                })?,
            };
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
            let ops = model_ops::ModelOperations::new();
            ops.show(&model)?;
        }
        Commands::Remove { model, force } => {
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
            let ops = model_ops::ModelOperations::new();
            ops.copy(&source, &destination)?;
        }
        Commands::Info { model } => {
            let ops = model_ops::ModelOperations::new();
            ops.info(&model)?;
        }
        Commands::Verify { model } => {
            let ops = model_ops::ModelOperations::new();
            ops.verify(&model)?;
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
                formatter.print_info(&format!(
                    "Set max cached models to {} (requires restart)",
                    max_models
                ));
            }
        }
        Commands::Config => {
            let formatter = output::OutputFormatter::new();
            formatter.print_header("Configuration Settings");

            use std::io::Write;
            let mut stdout = std::io::stdout();

            let model_path = config::get_model_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "Not set".to_string());
            let output_dir = config::get_output_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "./models".to_string());
            let top_k = config::get_top_k()
                .map(|k| k.to_string())
                .unwrap_or_else(|| "Not set".to_string());

            let config_table = format!(
                r#"
### Model Settings
- **Model Path:** `{}`
- **Output Directory:** `{}`
- **Mirror URL:** `{}`

### Generation Parameters
- **Temperature:** `{}`
- **Top-P:** `{}`
- **Top-K:** `{}`
- **Repeat Penalty:** `{}`
- **Max Tokens:** `{}`

### Device Settings
- **Device:** `{}`
- **Device Index:** `{}`

### Server Settings
- **Port:** `{}`

### Environment Variables
Set these in your `.env` file or environment:
- `MODEL_RS_MODEL_PATH` - Default model path
- `MODEL_RS_OUTPUT_DIR` - Download output directory
- `MODEL_RS_MIRROR` - HuggingFace mirror URL
- `MODEL_RS_TEMPERATURE` - Generation temperature
- `MODEL_RS_TOP_P` - Top-p sampling threshold
- `MODEL_RS_TOP_K` - Top-k sampling limit
- `MODEL_RS_REPEAT_PENALTY` - Repetition penalty
- `MODEL_RS_MAX_TOKENS` - Maximum tokens to generate
- `MODEL_RS_DEVICE` - Compute device (auto/cpu/metal/cuda)
- `MODEL_RS_DEVICE_INDEX` - GPU device index
- `MODEL_RS_PORT` - Server port
- `MODEL_RS_WARMUP_TOKENS` - Metal decode warmup passes (optional tuning)
"#,
                model_path,
                output_dir,
                config::get_mirror(),
                config::get_temperature(),
                config::get_top_p(),
                top_k,
                config::get_repeat_penalty(),
                config::get_max_tokens(),
                config::get_device(),
                config::get_device_index(),
                config::get_port(),
            );

            for chunk in config_table.split("\n\n") {
                formatter.print_markdown(chunk);
                stdout.flush()?;
            }
        }
    }

    Ok(())
}
