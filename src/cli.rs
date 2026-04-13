use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "model-rs")]
#[command(about = "Download and serve LLM models locally", long_about = None)]
#[command(version = concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION")))]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Download model from HuggingFace mirror")]
    #[command(visible_alias = "pull")]
    Download {
        #[arg(help = "Model name (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')")]
        model: String,

        #[arg(short = 'r', long, help = "Mirror URL (default: hf-mirror.com)")]
        mirror: Option<String>,

        #[arg(short, long, help = "Output directory")]
        output: Option<PathBuf>,
    },

    #[command(about = "Search for models on HuggingFace")]
    Search {
        #[arg(help = "Search query")]
        query: String,

        #[arg(short, long, default_value = "20", help = "Maximum number of results")]
        limit: usize,

        #[arg(short, long, help = "Filter by author/organization")]
        author: Option<String>,
    },

    #[command(about = "Serve local LLM over a web API")]
    Serve {
        #[arg(short, long, help = "Path to model directory")]
        model_path: Option<PathBuf>,

        #[arg(short, long, default_value = "8080", help = "Port to serve on")]
        port: u16,

        #[arg(
            long,
            default_value = "auto",
            help = "Compute device: auto|cpu|metal|cuda|mlx"
        )]
        device: String,

        #[arg(
            long,
            default_value = "0",
            help = "Device index (GPU ordinal) when using metal/cuda"
        )]
        device_index: usize,
    },

    #[command(about = "Generate text using the LLM")]
    Generate {
        #[arg(help = "Prompt text")]
        prompt: String,

        #[arg(long, help = "Optional system prompt")]
        system: Option<String>,

        #[arg(short, long, help = "Path to model directory")]
        model_path: Option<PathBuf>,

        #[arg(long, default_value = "512", help = "Maximum tokens to generate")]
        max_tokens: usize,

        #[arg(long, default_value = "0.7", help = "Temperature for generation")]
        temperature: f32,

        #[arg(
            long,
            default_value = "0.9",
            help = "Top-p (nucleus) sampling threshold"
        )]
        top_p: f32,

        #[arg(long, help = "Top-k sampling limit (default: disabled)")]
        top_k: Option<usize>,

        #[arg(long, default_value = "1.1", help = "Repetition penalty")]
        repeat_penalty: f32,

        #[arg(
            long,
            default_value = "auto",
            help = "Compute device: auto|cpu|metal|cuda|mlx"
        )]
        device: String,

        #[arg(
            long,
            default_value = "0",
            help = "Device index (GPU ordinal) when using metal/cuda"
        )]
        device_index: usize,
    },

    #[command(about = "Run model interactively (Ollama-style alias for chat)")]
    Run {
        #[arg(help = "Model name (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")]
        model: Option<String>,

        #[arg(long, help = "System prompt to set conversation context")]
        system: Option<String>,

        #[arg(
            long,
            default_value = "512",
            help = "Maximum tokens to generate per response"
        )]
        max_tokens: usize,

        #[arg(long, default_value = "0.7", help = "Temperature for generation")]
        temperature: f32,

        #[arg(
            long,
            default_value = "0.9",
            help = "Top-p (nucleus) sampling threshold"
        )]
        top_p: f32,

        #[arg(long, help = "Top-k sampling limit (default: disabled)")]
        top_k: Option<usize>,

        #[arg(long, default_value = "1.1", help = "Repetition penalty")]
        repeat_penalty: f32,

        #[arg(
            long,
            default_value = "auto",
            help = "Compute device: auto|cpu|metal|cuda|mlx"
        )]
        device: String,

        #[arg(
            long,
            default_value = "0",
            help = "Device index (GPU ordinal) when using metal/cuda"
        )]
        device_index: usize,

        #[arg(long, help = "Load chat session from file on startup")]
        session: Option<PathBuf>,

        #[arg(long, help = "Auto-save session to file on exit")]
        save_on_exit: Option<PathBuf>,
    },

    #[command(about = "Stop running model servers")]
    Stop {
        #[arg(help = "Model name/path to stop (Ollama-style)")]
        model: Option<String>,

        #[arg(long, help = "Port to stop server on")]
        port: Option<u16>,

        #[arg(short, long, help = "Force kill without confirmation")]
        force: bool,
    },

    #[command(about = "Interactive chat mode with conversation history and slash commands")]
    Chat {
        #[arg(short, long, help = "Path to model directory")]
        model_path: PathBuf,

        #[arg(long, help = "System prompt to set conversation context")]
        system: Option<String>,

        #[arg(
            long,
            default_value = "512",
            help = "Maximum tokens to generate per response"
        )]
        max_tokens: usize,

        #[arg(long, default_value = "0.7", help = "Temperature for generation")]
        temperature: f32,

        #[arg(
            long,
            default_value = "0.9",
            help = "Top-p (nucleus) sampling threshold"
        )]
        top_p: f32,

        #[arg(long, help = "Top-k sampling limit (default: disabled)")]
        top_k: Option<usize>,

        #[arg(long, default_value = "1.1", help = "Repetition penalty")]
        repeat_penalty: f32,

        #[arg(
            long,
            default_value = "auto",
            help = "Compute device: auto|cpu|metal|cuda|mlx"
        )]
        device: String,

        #[arg(
            long,
            default_value = "0",
            help = "Device index (GPU ordinal) when using metal/cuda"
        )]
        device_index: usize,

        #[arg(long, help = "Load chat session from file on startup")]
        session: Option<PathBuf>,

        #[arg(long, help = "Auto-save session to file on exit")]
        save_on_exit: Option<PathBuf>,
    },

    #[command(about = "List all downloaded models")]
    #[command(visible_alias = "ls")]
    List {
        #[arg(short, long, help = "Custom models directory")]
        models_dir: Option<PathBuf>,
    },

    #[command(about = "Deploy a model by starting the API server")]
    Deploy {
        #[arg(short, long, help = "Path to model directory")]
        model_path: Option<PathBuf>,

        #[arg(short, long, default_value = "8080", help = "Port to serve on")]
        port: u16,

        #[arg(
            long,
            default_value = "auto",
            help = "Compute device: auto|cpu|metal|cuda|mlx"
        )]
        device: String,

        #[arg(
            long,
            default_value = "0",
            help = "Device index (GPU ordinal) when using metal/cuda"
        )]
        device_index: usize,

        #[arg(long, help = "Run in background (detach from terminal)")]
        detached: bool,
    },

    #[command(about = "Generate embeddings for encoder-only models (BERT family)")]
    Embed {
        #[arg(help = "Input text")]
        text: String,

        #[arg(short, long, help = "Path to model directory")]
        model_path: PathBuf,

        #[arg(
            long,
            default_value = "auto",
            help = "Compute device: auto|cpu|metal|cuda|mlx"
        )]
        device: String,

        #[arg(
            long,
            default_value = "0",
            help = "Device index (GPU ordinal) when using metal/cuda"
        )]
        device_index: usize,
    },

    #[command(about = "Show current configuration settings")]
    Config,

    #[command(about = "Display detailed model information")]
    Show {
        #[arg(help = "Model name or path")]
        model: String,
    },

    #[command(about = "Remove a downloaded model from disk")]
    #[command(visible_alias = "rm")]
    Remove {
        #[arg(help = "Model name or path to remove")]
        model: String,

        #[arg(short, long, help = "Skip confirmation prompt")]
        force: bool,
    },

    #[command(about = "Show running model servers")]
    Ps,

    #[command(about = "Create a copy of a model")]
    Copy {
        #[arg(help = "Source model name or path")]
        source: String,

        #[arg(help = "Destination model name or path")]
        destination: String,
    },

    #[command(about = "Show detailed model metadata and capabilities")]
    Info {
        #[arg(help = "Model name or path")]
        model: String,
    },

    #[command(about = "Verify model integrity with checksum validation")]
    Verify {
        #[arg(help = "Model name or path to verify")]
        model: String,
    },

    #[command(about = "Manage in-memory model cache")]
    Cache {
        #[arg(short = 's', long, help = "Show cache statistics")]
        stats: bool,

        #[arg(short = 'c', long, help = "Clear the cache")]
        clear: bool,

        #[arg(short = 'e', long, help = "Enable or disable caching")]
        enable: Option<bool>,

        #[arg(short = 'p', long, help = "Preload a model into cache")]
        preload: Option<String>,

        #[arg(short = 'v', long, help = "Evict a model from cache")]
        evict: Option<String>,

        #[arg(long, help = "Set maximum cached models (requires restart)")]
        max: Option<usize>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_command_parsing() {
        let args = vec![
            "model-rs",
            "download",
            "test/model",
            "-r",
            "https://example.com",
            "-o",
            "/tmp/models",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Download {
                model,
                mirror,
                output,
            } => {
                assert_eq!(model, "test/model");
                assert_eq!(mirror, Some("https://example.com".to_string()));
                assert_eq!(output, Some(PathBuf::from("/tmp/models")));
            }
            _ => panic!("Expected Download command"),
        }
    }

    #[test]
    fn test_search_command_parsing() {
        let args = vec![
            "model-rs", "search", "llama", "--limit", "10", "--author", "meta",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Search {
                query,
                limit,
                author,
            } => {
                assert_eq!(query, "llama");
                assert_eq!(limit, 10);
                assert_eq!(author, Some("meta".to_string()));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_default_limit() {
        let args = vec!["model-rs", "search", "query"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Search { limit, .. } => {
                assert_eq!(limit, 20);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_generate_command_parsing() {
        let args = vec![
            "model-rs",
            "generate",
            "Hello world",
            "--system",
            "You are a helpful assistant.",
            "--model-path",
            "/models",
            "--max-tokens",
            "100",
            "--temperature",
            "0.5",
            "--top-p",
            "0.8",
            "--top-k",
            "50",
            "--repeat-penalty",
            "1.2",
            "--device",
            "cpu",
            "--device-index",
            "1",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
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
                assert_eq!(prompt, "Hello world");
                assert_eq!(system, Some("You are a helpful assistant.".to_string()));
                assert_eq!(model_path, Some(PathBuf::from("/models")));
                assert_eq!(max_tokens, 100);
                assert_eq!(temperature, 0.5);
                assert_eq!(top_p, 0.8);
                assert_eq!(top_k, Some(50));
                assert_eq!(repeat_penalty, 1.2);
                assert_eq!(device, "cpu");
                assert_eq!(device_index, 1);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_default_values() {
        let args = vec!["model-rs", "generate", "test"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Generate {
                max_tokens,
                temperature,
                device,
                device_index,
                ..
            } => {
                assert_eq!(max_tokens, 512);
                assert_eq!(temperature, 0.7);
                assert_eq!(device, "auto");
                assert_eq!(device_index, 0);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_serve_command_parsing() {
        let args = vec![
            "model-rs",
            "serve",
            "--model-path",
            "/models",
            "--port",
            "9000",
            "--device",
            "cpu",
            "--device-index",
            "1",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve {
                model_path,
                port,
                device,
                device_index,
            } => {
                assert_eq!(model_path, Some(PathBuf::from("/models")));
                assert_eq!(port, 9000);
                assert_eq!(device, "cpu");
                assert_eq!(device_index, 1);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_serve_default_values() {
        let args = vec!["model-rs", "serve"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve {
                port,
                device,
                device_index,
                ..
            } => {
                assert_eq!(port, 8080);
                assert_eq!(device, "auto");
                assert_eq!(device_index, 0);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_invalid_command() {
        let args = vec!["model-rs", "invalid-command"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_err());
    }

    #[test]
    fn test_chat_command_parsing() {
        let args = vec![
            "model-rs",
            "chat",
            "--model-path",
            "/models",
            "--system",
            "You are a helpful assistant.",
            "--max-tokens",
            "256",
            "--temperature",
            "0.8",
            "--top-p",
            "0.85",
            "--top-k",
            "40",
            "--repeat-penalty",
            "1.15",
            "--device",
            "metal",
            "--device-index",
            "1",
            "--session",
            "chat.json",
            "--save-on-exit",
            "output.json",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
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
                assert_eq!(model_path, PathBuf::from("/models"));
                assert_eq!(system, Some("You are a helpful assistant.".to_string()));
                assert_eq!(max_tokens, 256);
                assert_eq!(temperature, 0.8);
                assert_eq!(top_p, 0.85);
                assert_eq!(top_k, Some(40));
                assert_eq!(repeat_penalty, 1.15);
                assert_eq!(device, "metal");
                assert_eq!(device_index, 1);
                assert_eq!(session, Some(PathBuf::from("chat.json")));
                assert_eq!(save_on_exit, Some(PathBuf::from("output.json")));
            }
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn test_chat_default_values() {
        let args = vec!["model-rs", "chat", "--model-path", "/models"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Chat {
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                device,
                device_index,
                session,
                save_on_exit,
                ..
            } => {
                assert_eq!(max_tokens, 512);
                assert_eq!(temperature, 0.7);
                assert_eq!(top_p, 0.9);
                assert_eq!(top_k, None);
                assert_eq!(repeat_penalty, 1.1);
                assert_eq!(device, "auto");
                assert_eq!(device_index, 0);
                assert_eq!(session, None);
                assert_eq!(save_on_exit, None);
            }
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn test_chat_requires_model_path() {
        let args = vec!["model-rs", "chat", "--temperature", "0.5"];
        let cli = Cli::try_parse_from(args);

        // Should fail because model-path is required
        assert!(cli.is_err());
    }

    #[test]
    fn test_generate_with_all_sampling_params() {
        let args = vec![
            "model-rs",
            "generate",
            "Hello",
            "--model-path",
            "/models",
            "--temperature",
            "0.6",
            "--top-p",
            "0.92",
            "--top-k",
            "50",
            "--repeat-penalty",
            "1.05",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Generate {
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                ..
            } => {
                assert_eq!(temperature, 0.6);
                assert_eq!(top_p, 0.92);
                assert_eq!(top_k, Some(50));
                assert_eq!(repeat_penalty, 1.05);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_embed_command_parsing() {
        let args = vec![
            "model-rs",
            "embed",
            "Hello world",
            "--model-path",
            "/models",
            "--device",
            "cpu",
            "--device-index",
            "1",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Embed {
                text,
                model_path,
                device,
                device_index,
            } => {
                assert_eq!(text, "Hello world");
                assert_eq!(model_path, PathBuf::from("/models"));
                assert_eq!(device, "cpu");
                assert_eq!(device_index, 1);
            }
            _ => panic!("Expected Embed command"),
        }
    }

    #[test]
    fn test_embed_default_values() {
        let args = vec!["model-rs", "embed", "x", "--model-path", "/models"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Embed {
                device,
                device_index,
                ..
            } => {
                assert_eq!(device, "auto");
                assert_eq!(device_index, 0);
            }
            _ => panic!("Expected Embed command"),
        }
    }

    #[test]
    fn test_show_command_parsing() {
        let args = vec!["model-rs", "show", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Show { model } => {
                assert_eq!(model, "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
            }
            _ => panic!("Expected Show command"),
        }
    }

    #[test]
    fn test_remove_command_parsing() {
        let args = vec!["model-rs", "remove", "model-name", "--force"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Remove { model, force } => {
                assert_eq!(model, "model-name");
                assert!(force);
            }
            _ => panic!("Expected Remove command"),
        }
    }

    #[test]
    fn test_rm_alias() {
        let args = vec!["model-rs", "rm", "model-name"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Remove { model, force } => {
                assert_eq!(model, "model-name");
                assert!(!force);
            }
            _ => panic!("Expected Remove command"),
        }
    }

    #[test]
    fn test_ps_command_parsing() {
        let args = vec!["model-rs", "ps"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Ps => {}
            _ => panic!("Expected Ps command"),
        }
    }

    #[test]
    fn test_copy_command_parsing() {
        let args = vec!["model-rs", "copy", "source-model", "dest-model"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Copy {
                source,
                destination,
            } => {
                assert_eq!(source, "source-model");
                assert_eq!(destination, "dest-model");
            }
            _ => panic!("Expected Copy command"),
        }
    }

    #[test]
    fn test_info_command_parsing() {
        let args = vec!["model-rs", "info", "model-name"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Info { model } => {
                assert_eq!(model, "model-name");
            }
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_verify_command_parsing() {
        let args = vec!["model-rs", "verify", "model-name"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Verify { model } => {
                assert_eq!(model, "model-name");
            }
            _ => panic!("Expected Verify command"),
        }
    }

    #[test]
    fn test_run_command_parsing() {
        let args = vec![
            "model-rs",
            "run",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--system",
            "You are helpful.",
            "--max-tokens",
            "256",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Run {
                model,
                system,
                max_tokens,
                ..
            } => {
                assert_eq!(
                    model,
                    Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string())
                );
                assert_eq!(system, Some("You are helpful.".to_string()));
                assert_eq!(max_tokens, 256);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_run_command_default_model() {
        let args = vec![
            "model-rs",
            "run",
            "--system",
            "You are helpful.",
            "--max-tokens",
            "256",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Run {
                model,
                system,
                max_tokens,
                ..
            } => {
                assert_eq!(model, None);
                assert_eq!(system, Some("You are helpful.".to_string()));
                assert_eq!(max_tokens, 256);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_stop_command_parsing() {
        let args = vec![
            "model-rs",
            "stop",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--port",
            "8080",
            "--force",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Stop { model, port, force } => {
                assert_eq!(
                    model,
                    Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string())
                );
                assert_eq!(port, Some(8080));
                assert!(force);
            }
            _ => panic!("Expected Stop command"),
        }
    }

    #[test]
    fn test_pull_alias() {
        let args = vec!["model-rs", "pull", "test/model"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Download { model, .. } => {
                assert_eq!(model, "test/model");
            }
            _ => panic!("Expected Download command (pull alias)"),
        }
    }

    #[test]
    fn test_ls_alias() {
        let args = vec!["model-rs", "ls"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::List { .. } => {}
            _ => panic!("Expected List command (ls alias)"),
        }
    }
}
