use crate::error::{ModelError, Result};
use crate::local::{LocalModel, LocalModelConfig};
use crate::local::DevicePreference;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tracing::info;
use serde::{Deserialize, Serialize};

mod service;
mod server;

pub use service::LlmService;

/// Chat session for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatSession {
    model_path: PathBuf,
    system_prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    messages: Vec<ChatMessage>,
    created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String, // "system", "user", "assistant"
    content: String,
    timestamp: String,
}

impl ChatSession {
    fn new(
        model_path: &Path,
        system_prompt: Option<&str>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
    ) -> Self {
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            if !system.trim().is_empty() {
                messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: system.trim().to_string(),
                    timestamp: now.clone(),
                });
            }
        }

        Self {
            model_path: model_path.to_path_buf(),
            system_prompt: system_prompt.map(|s| s.trim().to_string()),
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            messages,
            created_at: now,
        }
    }

    fn add_message(&mut self, role: &str, content: &str) {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp,
        });
    }

    fn get_conversation_history(&self) -> String {
        self.messages
            .iter()
            .map(|msg| {
                let role = if msg.role == "system" {
                    "System"
                } else if msg.role == "user" {
                    "User"
                } else {
                    "Assistant"
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn truncate_history(&mut self, keep_last_n: usize) {
        // Keep system prompt if exists
        let has_system = self.messages.first().map_or(false, |m| m.role == "system");

        if has_system && self.messages.len() > keep_last_n {
            let system_msg = self.messages[0].clone();
            let total_msgs = self.messages.len();
            let max_keep = keep_last_n - 1;
            let start = if total_msgs > max_keep { total_msgs - max_keep } else { 1 };

            let remaining: Vec<_> = self.messages.drain(..).skip(start).collect();
            self.messages = vec![system_msg];
            self.messages.extend(remaining);
        } else if self.messages.len() > keep_last_n {
            let start = self.messages.len() - keep_last_n;
            self.messages = self.messages.drain(..).skip(start).collect();
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ModelError::JsonError(e))?;
        fs::write(path, json)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to save session: {}", e)))?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to load session: {}", e)))?;
        let session: ChatSession = serde_json::from_str(&content)
            .map_err(|e| ModelError::JsonError(e))?;
        Ok(session)
    }
}

/// Parse and execute slash commands
fn handle_command(
    input: &str,
    session: &mut ChatSession,
    local_model: &mut LocalModel,
) -> Result<bool> {
    let parts: Vec<&str> = input.trim().splitn(3, ' ').collect();
    let cmd = parts.get(0).map(|s| s.trim()).unwrap_or("");

    if !cmd.starts_with('/') {
        return Ok(false); // Not a command
    }

    let formatter = crate::output::OutputFormatter::new();
    
    match cmd {
        "/help" => {
            formatter.print_help_commands();
        }
        "/clear" => {
            formatter.print_success("Conversation history cleared");
            local_model.clear_session_kv_cache();
            if let Some(system) = &session.system_prompt {
                session.messages = vec![ChatMessage {
                    role: "system".to_string(),
                    content: system.clone(),
                    timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                }];
            } else {
                session.messages.clear();
            }
        }
        "/history" => {
            formatter.print_section("Conversation History", &format!("{} messages", session.messages.len()));
            for (idx, msg) in session.messages.iter().enumerate() {
                let role_icon = if msg.role == "system" {
                    "🔧"
                } else if msg.role == "user" {
                    "👤"
                } else {
                    "🤖"
                };
                formatter.print_markdown(&format!("{}. {} **[{}]:** {}\n", idx + 1, role_icon, msg.role, msg.content));
            }
        }
        "/search" => {
            let query = parts.get(1).map(|s| s.trim()).unwrap_or("");
            if query.is_empty() {
                formatter.print_error("Usage: /search <query>");
                formatter.print_markdown("**Example:** `/search temperature`\n");
                return Ok(true);
            }

            let query_lower = query.to_lowercase();
            formatter.print_section(
                "Search Results",
                &format!("Query: `{}`", query),
            );

            let mut found = 0usize;
            for (idx, msg) in session.messages.iter().enumerate() {
                if msg.content.to_lowercase().contains(&query_lower) {
                    found += 1;
                    let role_icon = if msg.role == "system" {
                        "🔧"
                    } else if msg.role == "user" {
                        "👤"
                    } else {
                        "🤖"
                    };

                    formatter.print_markdown(&format!(
                        "{}. {} **[{}]:** {}\n",
                        idx + 1,
                        role_icon,
                        msg.role,
                        msg.content
                    ));
                }
            }

            if found == 0 {
                formatter.print_info("No matching messages found");
            }
        }
        "/save" => {
            let filename = parts.get(1).unwrap_or(&"");
            if filename.is_empty() {
                formatter.print_error("Usage: /save <filename>");
                formatter.print_markdown("**Example:** `/save my_chat.json`\n");
                return Ok(true);
            }
            let path = PathBuf::from(filename);
            session.save(&path)?;
            formatter.print_success(&format!("Conversation saved to: {}", filename));
        }
        "/load" => {
            let filename = parts.get(1).unwrap_or(&"");
            if filename.is_empty() {
                formatter.print_error("Usage: /load <filename>");
                formatter.print_markdown("**Example:** `/load my_chat.json`\n");
                return Ok(true);
            }
            let path = PathBuf::from(filename);
            let loaded_session = ChatSession::load(&path)?;
            formatter.print_success(&format!("Loaded conversation from: {}", filename));
            formatter.print_info(&format!("Messages: {}", loaded_session.messages.len()));
            *session = loaded_session;
            local_model.clear_session_kv_cache();
        }
        "/set" => {
            let param = parts.get(1).map(|s| s.trim()).unwrap_or("");
            let value = parts.get(2).map(|s| s.trim()).unwrap_or("");

            if param.is_empty() || value.is_empty() {
                formatter.print_error("Usage: /set <parameter> <value>");
                formatter.print_markdown("**Available parameters:** temperature, top_p, top_k, repeat_penalty, max_tokens\n");
                formatter.print_markdown("**Example:** `/set temperature 0.8`\n");
                return Ok(true);
            }

            match param {
                "temperature" => {
                    let new_temp = value.parse::<f32>()
                        .map_err(|_| ModelError::InvalidConfig("Invalid temperature value".to_string()))?;
                    if !(0.0..=2.0).contains(&new_temp) {
                        formatter.print_error("Temperature must be between 0.0 and 2.0");
                        return Ok(true);
                    }
                    session.temperature = new_temp;
                    formatter.print_success(&format!("Temperature set to {}", new_temp));
                }
                "top_p" => {
                    let new_top_p = value.parse::<f32>()
                        .map_err(|_| ModelError::InvalidConfig("Invalid top_p value".to_string()))?;
                    if !(0.0..=1.0).contains(&new_top_p) {
                        formatter.print_error("Top-p must be between 0.0 and 1.0");
                        return Ok(true);
                    }
                    session.top_p = new_top_p;
                    formatter.print_success(&format!("Top-p set to {}", new_top_p));
                }
                "top_k" => {
                    let new_top_k = value.parse::<usize>()
                        .map_err(|_| ModelError::InvalidConfig("Invalid top_k value".to_string()))?;
                    session.top_k = Some(new_top_k);
                    formatter.print_success(&format!("Top-k set to {}", new_top_k));
                }
                "repeat_penalty" => {
                    let new_rp = value.parse::<f32>()
                        .map_err(|_| ModelError::InvalidConfig("Invalid repeat_penalty value".to_string()))?;
                    if !(0.0..=2.0).contains(&new_rp) {
                        formatter.print_error("Repeat penalty must be between 0.0 and 2.0");
                        return Ok(true);
                    }
                    session.repeat_penalty = new_rp;
                    formatter.print_success(&format!("Repeat penalty set to {}", new_rp));
                }
                "max_tokens" => {
                    let new_max = value.parse::<usize>()
                        .map_err(|_| ModelError::InvalidConfig("Invalid max_tokens value".to_string()))?;
                    session.max_tokens = new_max;
                    formatter.print_success(&format!("Max tokens set to {}", new_max));
                }
                _ => {
                    formatter.print_error(&format!("Unknown parameter: {}", param));
                    formatter.print_markdown("**Available:** temperature, top_p, top_k, repeat_penalty, max_tokens\n");
                }
            }
        }
        "/quit" | "/exit" => {
            return Err(ModelError::LocalModelError("Exiting chat".to_string()));
        }
        _ => {
            formatter.print_error(&format!("Unknown command: {}", cmd));
            formatter.print_markdown("Type `/help` for available commands\n");
        }
    }

    Ok(true) // Command was handled
}

/// Serve local LLM over a web API (REST + SSE)
pub async fn serve(
    model_path: Option<&Path>,
    port: u16,
    device: &str,
    device_index: usize,
) -> Result<()> {
    server::serve(model_path, port, device, device_index).await
}

/// Generate text using a local LLM model
pub async fn generate(
    prompt: &str,
    system: Option<&str>,
    model_path: Option<&Path>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    device: &str,
    device_index: usize,
) -> Result<()> {
    info!("Generating response for prompt: {}", prompt);

    // Require local model path
    let path = model_path.ok_or_else(|| ModelError::InvalidConfig(
        "Model path is required for generation. Use --model-path <path> to specify a local model directory.".to_string()
    ))?;

    info!("Using local model from: {}", path.display());

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: path.to_path_buf(),
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        max_seq_len: max_tokens * 2, // Give some room for the prompt
        device_preference,
        device_index,
        ..Default::default()
    };

    let mut local_model = LocalModel::load(config).await?;
    local_model.enable_session_kv_cache();
    let formatter = crate::output::OutputFormatter::new();
    let mut stream_md = crate::output::MarkdownStreamRenderer::new();
    formatter.print_header("Local Generation");

    let effective_prompt = match system {
        Some(system_prompt) if !system_prompt.trim().is_empty() => {
            format!(
                "System: {}\n\nUser: {}\n\nAssistant:",
                system_prompt.trim(),
                prompt
            )
        }
        _ => {
            // Always use chat format for better compatibility with chat-tuned models
            format!("User: {}\n\nAssistant:", prompt)
        }
    };

    // Generate with early stopping if model starts a new turn
    use std::sync::atomic::{AtomicBool, Ordering};
    let should_stop = AtomicBool::new(false);

    let mut code_hi: Option<crate::output::CodeHighlighter<'_>> = None;
    let mut code_open = false;

    local_model.generate_stream_with(&effective_prompt, max_tokens, temperature, |piece| {
        if should_stop.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Check if this piece or the cumulative buffer contains "User:" indicating a new turn
        if piece.contains("\nUser:") || piece.contains("\n\nUser:") {
            should_stop.store(true, Ordering::Relaxed);
            return Ok(());
        }

        stream_md.push_with(
            &piece,
            |text| {
                print!("{}", text);
                let _ = io::stdout().flush();
            },
            |ev| {
                match ev {
                    crate::output::CodeStreamEvent::Start { language } => {
                        if !code_open {
                            println!();
                            code_open = true;
                        }
                        code_hi = Some(formatter.code_highlighter(language));
                    }
                    crate::output::CodeStreamEvent::Chunk { language: _, code } => {
                        if let Some(h) = code_hi.as_mut() {
                            h.write(code);
                        }
                    }
                    crate::output::CodeStreamEvent::End => {
                        if let Some(h) = code_hi.as_mut() {
                            h.finish_line();
                        }
                        code_hi = None;
                        code_open = false;
                        println!();
                    }
                }
            },
        );

        Ok(())
    }).await?;

    stream_md.finish_with(
        |text| {
            print!("{}", text);
            let _ = io::stdout().flush();
        },
        |ev| {
            match ev {
                crate::output::CodeStreamEvent::Start { language } => {
                    if !code_open {
                        println!();
                        code_open = true;
                    }
                    code_hi = Some(formatter.code_highlighter(language));
                }
                crate::output::CodeStreamEvent::Chunk { language: _, code } => {
                    if let Some(h) = code_hi.as_mut() {
                        h.write(code);
                    }
                }
                crate::output::CodeStreamEvent::End => {
                    if let Some(h) = code_hi.as_mut() {
                        h.finish_line();
                    }
                    code_hi = None;
                    code_open = false;
                    println!();
                }
            }
        },
    );

    println!();
    
    formatter.print_success("Generation complete");

    Ok(())
}

pub async fn embed(text: &str, model_path: &Path, device: &str, device_index: usize) -> Result<()> {
    info!("Generating embedding");

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: model_path.to_path_buf(),
        device_preference,
        device_index,
        ..Default::default()
    };

    let mut local_model = LocalModel::load(config).await?;
    let embedding = local_model.embed_text(text).await?;

    println!("{}", serde_json::to_string(&embedding).map_err(|e| ModelError::JsonError(e))?);
    
    Ok(())
}

/// Interactive chat mode with conversation history and slash commands
pub async fn chat(
    model_path: &Path,
    system: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    device: &str,
    device_index: usize,
    session_file: Option<&Path>,
    save_on_exit: Option<&Path>,
) -> Result<()> {
    info!("Starting interactive chat mode");

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: model_path.to_path_buf(),
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        max_seq_len: 4096, // Fixed max context length for chat
        device_preference,
        device_index,
        ..Default::default()
    };

    let mut local_model = LocalModel::load(config).await?;
    let formatter = crate::output::OutputFormatter::new();

    formatter.print_chat_header();

    // Initialize or load chat session
    let mut session = if let Some(session_path) = session_file {
        match ChatSession::load(session_path) {
            Ok(loaded_session) => {
                formatter.print_success(&format!("Loaded session from: {}", session_path.display()));
                formatter.print_info(&format!("Messages: {}", loaded_session.messages.len()));
                loaded_session
            }
            Err(e) => {
                formatter.print_warning(&format!("Failed to load session: {}. Starting new session.", e));
                ChatSession::new(
                    model_path,
                    system,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                )
            }
        }
    } else {
        ChatSession::new(
            model_path,
            system,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
        )
    };

    loop {
        // Print user prompt
        print!("You: ");
        io::stdout().flush()?;

        // Read user input
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to read input: {}", e)))?;

        // Support Ollama-style multiline prompts using triple quotes: `""" ... """`
        // Example:
        //   You: """
        //   <multi line content>
        //   """
        let mut user_input = user_input.trim_end_matches(&['\n', '\r'][..]).to_string();
        let trimmed = user_input.trim_start();
        if trimmed.starts_with("\"\"\"") {
            // If opening and closing markers are on the same line, just extract.
            if trimmed.ends_with("\"\"\"") && trimmed.len() >= 6 {
                let inner = &trimmed[3..trimmed.len() - 3];
                user_input = inner.trim().to_string();
            } else {
                // Collect until we see a line that contains the closing marker.
                let mut collected = trimmed.trim_start_matches("\"\"\"").to_string();
                collected = collected.trim_start().to_string();

                loop {
                    let mut next = String::new();
                    io::stdin().read_line(&mut next)
                        .map_err(|e| ModelError::LocalModelError(format!("Failed to read input: {}", e)))?;

                    let next_trim_end = next.trim_end_matches(&['\n', '\r'][..]);
                    let next_trim = next_trim_end.trim();

                    if next_trim.ends_with("\"\"\"") && next_trim.len() >= 3 {
                        let before = next_trim_end.trim_end_matches("\"\"\"");
                        if !before.is_empty() {
                            collected.push_str(before.trim_end());
                        }
                        break;
                    } else {
                        collected.push_str(next_trim_end);
                        collected.push('\n');
                    }
                }

                user_input = collected.trim().to_string();
            }
        }
        let user_input = user_input.trim();

        // Check for exit commands (also handled by /quit, but keeping for consistency)
        if user_input.eq_ignore_ascii_case("quit") || user_input.eq_ignore_ascii_case("exit") {
            if let Some(save_path) = save_on_exit {
                session.save(save_path)?;
                formatter.print_success(&format!("Session saved to: {}", save_path.display()));
            }
            formatter.print_info("Goodbye!");
            break;
        }

        // Skip empty input
        if user_input.is_empty() {
            continue;
        }

        // Check for slash commands
        if user_input.starts_with('/') {
            match handle_command(user_input, &mut session, &mut local_model) {
                Ok(handled) => {
                    if handled {
                        continue; // Command was handled, don't generate
                    }
                }
                Err(e) => {
                    if e.to_string() == "Exiting chat" {
                        if let Some(save_path) = save_on_exit {
                            session.save(save_path)?;
                            formatter.print_success(&format!("Session saved to: {}", save_path.display()));
                        }
                        formatter.print_info("Goodbye!");
                        break;
                    }
                    formatter.print_error(&format!("Error: {}", e));
                    continue;
                }
            }
        }

        // Add user message to session
        session.add_message("user", user_input);

        // Build prompt from conversation history
        let conversation_history = session.get_conversation_history();
        let full_prompt = format!("{}\nAssistant:", conversation_history);

        // Generate response
        print!("Assistant: ");
        io::stdout().flush()?;

        let response = local_model.generate_text(&full_prompt, session.max_tokens, session.temperature).await?;

        println!("{}", response);

        // Add assistant response to session
        session.add_message("assistant", &response);
        println!();

        // Prevent conversation from growing too large
        // Keep last 10 turns (20 messages: user + assistant pairs, plus system prompt)
        if session.messages.len() > 20 {
            session.truncate_history(20);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_session_new() {
        let session = ChatSession::new(
            Path::new("/models/test"),
            Some("You are a helpful assistant"),
            512,
            0.7,
            0.9,
            None,
            1.1,
        );

        assert_eq!(session.model_path, PathBuf::from("/models/test"));
        assert_eq!(session.system_prompt, Some("You are a helpful assistant".to_string()));
        assert_eq!(session.max_tokens, 512);
        assert_eq!(session.temperature, 0.7);
        assert_eq!(session.top_p, 0.9);
        assert_eq!(session.top_k, None);
        assert_eq!(session.repeat_penalty, 1.1);
        assert_eq!(session.messages.len(), 1); // System prompt
        assert_eq!(session.messages[0].role, "system");
    }

    #[test]
    fn test_chat_session_add_message() {
        let mut session = ChatSession::new(
            Path::new("/models/test"),
            None,
            512,
            0.7,
            0.9,
            None,
            1.1,
        );

        session.add_message("user", "Hello!");
        session.add_message("assistant", "Hi there!");

        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, "user");
        assert_eq!(session.messages[0].content, "Hello!");
        assert_eq!(session.messages[1].role, "assistant");
        assert_eq!(session.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_chat_session_get_conversation_history() {
        let mut session = ChatSession::new(
            Path::new("/models/test"),
            None,
            512,
            0.7,
            0.9,
            None,
            1.1,
        );

        session.add_message("user", "Hello!");
        session.add_message("assistant", "Hi there!");

        let history = session.get_conversation_history();
        assert!(history.contains("User: Hello!"));
        assert!(history.contains("Assistant: Hi there!"));
    }

    #[test]
    fn test_chat_session_truncate_history() {
        let mut session = ChatSession::new(
            Path::new("/models/test"),
            Some("System prompt"),
            512,
            0.7,
            0.9,
            None,
            1.1,
        );

        // Add 25 messages (more than the limit of 20)
        for i in 0..25 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            session.add_message(role, &format!("Message {}", i));
        }

        assert_eq!(session.messages.len(), 26); // 1 system + 25 messages

        // Truncate to 20
        session.truncate_history(20);

        // Should have system prompt + last 19 messages = 20 total
        assert_eq!(session.messages.len(), 20);
        assert_eq!(session.messages[0].role, "system");
        assert_eq!(session.messages[0].content, "System prompt");
    }

    #[test]
    fn test_chat_session_truncate_history_no_system() {
        let mut session = ChatSession::new(
            Path::new("/models/test"),
            None,
            512,
            0.7,
            0.9,
            None,
            1.1,
        );

        // Add 25 messages
        for i in 0..25 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            session.add_message(role, &format!("Message {}", i));
        }

        assert_eq!(session.messages.len(), 25);

        // Truncate to 20
        session.truncate_history(20);

        // Should have last 20 messages
        assert_eq!(session.messages.len(), 20);
    }

    #[test]
    fn test_chat_session_save_and_load() {
        let mut session = ChatSession::new(
            Path::new("/models/test"),
            Some("You are helpful"),
            256,
            0.8,
            0.85,
            Some(50),
            1.2,
        );

        session.add_message("user", "Test message");
        session.add_message("assistant", "Test response");

        // Test save
        let temp_file = PathBuf::from("/tmp/test_chat_session.json");
        let result = session.save(&temp_file);
        assert!(result.is_ok());

        // Test load
        let loaded_session = ChatSession::load(&temp_file).unwrap();
        assert_eq!(loaded_session.system_prompt, session.system_prompt);
        assert_eq!(loaded_session.max_tokens, session.max_tokens);
        assert_eq!(loaded_session.temperature, session.temperature);
        assert_eq!(loaded_session.top_p, session.top_p);
        assert_eq!(loaded_session.top_k, session.top_k);
        assert_eq!(loaded_session.repeat_penalty, session.repeat_penalty);
        assert_eq!(loaded_session.messages.len(), session.messages.len());

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_chat_session_serialize_deserialize() {
        let session = ChatSession::new(
            Path::new("/models/test"),
            Some("System prompt"),
            512,
            0.7,
            0.9,
            Some(40),
            1.1,
        );

        // Serialize
        let json = serde_json::to_string(&session);
        assert!(json.is_ok());

        // Deserialize
        let deserialized: std::result::Result<ChatSession, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());

        let loaded = deserialized.unwrap();
        assert_eq!(loaded.system_prompt, session.system_prompt);
        assert_eq!(loaded.temperature, session.temperature);
    }
}
