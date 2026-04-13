use crate::error::{ModelError, Result};
use crate::output::OutputFormatter;
use directories::ProjectDirs;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub struct ModelOperations {
    formatter: OutputFormatter,
}

impl ModelOperations {
    pub fn new() -> Self {
        Self {
            formatter: OutputFormatter::new(),
        }
    }

    fn get_models_dir() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "modelrs", "modelrs").ok_or_else(|| {
            ModelError::InvalidConfig("Could not determine home directory".to_string())
        })?;
        Ok(proj_dirs.cache_dir().join("models"))
    }

    pub fn resolve_model_path(&self, model: &str) -> Result<PathBuf> {
        let path = Path::new(model);

        if path.is_absolute() && path.exists() {
            return Ok(path.to_path_buf());
        }

        let models_dir = Self::get_models_dir()?;
        let normalized = model.replace('/', "--");
        let model_path = models_dir.join(&normalized);

        if model_path.exists() {
            return Ok(model_path);
        }

        Err(ModelError::ModelNotFound(format!(
            "Model not found: {}\nSearched in: {}",
            model,
            model_path.display()
        )))
    }

    pub fn show(&self, model: &str) -> Result<()> {
        let model_path = self.resolve_model_path(model)?;

        self.formatter.print_header(&format!("Model: {}", model));
        println!();

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(
                "config.json not found in model directory".to_string(),
            ));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let config: Value = serde_json::from_str(&config_content)?;

        let mut info = String::new();
        info.push_str(&format!("**Path:** `{}`\n\n", model_path.display()));

        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            info.push_str(&format!("**Architecture:** {}\n\n", model_type));
        }

        if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
            info.push_str(&format!("**Hidden Size:** {}\n\n", hidden_size));
        }

        if let Some(num_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            info.push_str(&format!("**Layers:** {}\n\n", num_layers));
        }

        if let Some(num_heads) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
            info.push_str(&format!("**Attention Heads:** {}\n\n", num_heads));
        }

        if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
            info.push_str(&format!("**Vocabulary Size:** {}\n\n", vocab_size));
        }

        if let Some(max_pos) = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            info.push_str(&format!("**Max Position Embeddings:** {}\n\n", max_pos));
        }

        let total_size = self.calculate_directory_size(&model_path)?;
        info.push_str(&format!(
            "**Total Size:** {}\n\n",
            Self::format_size(total_size)
        ));

        let file_count = self.count_files(&model_path)?;
        info.push_str(&format!("**Files:** {}\n\n", file_count));

        self.formatter.print_markdown(&info);

        let weight_files = self.find_weight_files(&model_path)?;
        if !weight_files.is_empty() {
            self.formatter.print_section("Weight Files", "");
            for file in weight_files {
                let file_size = fs::metadata(&file)?.len();
                println!(
                    "- {} ({})",
                    file.file_name().unwrap().to_string_lossy(),
                    Self::format_size(file_size)
                );
            }
            println!();
        }

        if let Some(template) = config.get("chat_template").and_then(|v| v.as_str()) {
            self.formatter
                .print_section("Chat Template", &format!("`{}`", template));
        }

        Ok(())
    }

    pub fn remove(&self, model: &str, force: bool) -> Result<()> {
        let model_path = self.resolve_model_path(model)?;

        if !force {
            print!(
                "Are you sure you want to remove model at '{}'? [y/N]: ",
                model_path.display()
            );
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Cancelled.");
                return Ok(());
            }
        }

        let total_size = self.calculate_directory_size(&model_path)?;

        fs::remove_dir_all(&model_path)?;

        println!("✓ Removed model: {}", model);
        println!("  Freed {} of disk space", Self::format_size(total_size));

        // Refresh cached model index (best-effort).
        if let Some(parent) = model_path.parent() {
            let _ = crate::models::refresh_models_index(Some(parent));
        }

        Ok(())
    }

    pub fn ps(&self) -> Result<()> {
        self.formatter.print_header("Running Model Servers");
        println!();

        println!("Note: This command shows process information for running Influence servers.");
        println!("Use 'ps aux | grep model-rs' for detailed process information.\n");

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let output = Command::new("ps").args(&["aux"]).output()?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut found = false;

            for line in stdout.lines() {
                if line.contains("model-rs serve") && !line.contains("grep") {
                    if !found {
                        println!("**Active Servers:**\n");
                        found = true;
                    }
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 11 {
                        let pid = parts[1];
                        let cpu = parts[2];
                        let mem = parts[3];
                        println!("- PID: {} | CPU: {}% | MEM: {}%", pid, cpu, mem);
                    }
                }
            }

            if !found {
                println!("No running Influence servers found.");
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            println!("Process listing not implemented for this platform.");
            println!("Use 'ps aux | grep \"model-rs serve\"' to see running servers.");
        }

        Ok(())
    }

    pub fn stop(&self, model: Option<&str>, port: Option<u16>, force: bool) -> Result<()> {
        if model.is_none() && port.is_none() {
            return Err(ModelError::InvalidConfig(
                "Provide either a model name/path or --port to stop a running server".to_string(),
            ));
        }

        #[cfg(target_os = "macos")]
        {
            let model_path = match model {
                Some(m) => Some(self.resolve_model_path(m)?),
                None => None,
            };

            use std::process::Command;

            let output = Command::new("ps").args(&["aux"]).output()?;
            let stdout = String::from_utf8_lossy(&output.stdout);

            let mut pids: Vec<u32> = Vec::new();
            for line in stdout.lines() {
                if !line.contains("model-rs serve") || line.contains("grep") {
                    continue;
                }

                let mut ok = true;

                if let Some(p) = port {
                    let needle = format!("--port {}", p);
                    ok &= line.contains(&needle) || line.contains(&format!(":{}", p));
                }

                if let Some(ref mp) = model_path {
                    let mp_str = mp.to_string_lossy();
                    ok &= !mp_str.is_empty() && line.contains(mp_str.as_ref());
                }

                if ok {
                    if let Some(pid_str) = line.split_whitespace().nth(1) {
                        if let Ok(pid) = pid_str.parse::<u32>() {
                            pids.push(pid);
                        }
                    }
                }
            }

            if pids.is_empty() {
                println!("No running model-rs servers matched the provided criteria.");
                return Ok(());
            }

            pids.sort_unstable();
            pids.dedup();

            if !force {
                print!(
                    "Stop {} server process(es)? [y/N]: ",
                    pids.len()
                );
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;

                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            for pid in &pids {
                let mut cmd = Command::new("kill");
                if force {
                    cmd.arg("-9");
                }
                cmd.arg(pid.to_string()).status()?;
            }

            println!("✓ Stopped {} server process(es).", pids.len());
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = model;
            let _ = port;
            println!("Stop is not implemented on this platform. Try killing the server PID manually.");
            Ok(())
        }
    }

    pub fn copy(&self, source: &str, destination: &str) -> Result<()> {
        let source_path = self.resolve_model_path(source)?;

        let models_dir = Self::get_models_dir()?;
        let dest_normalized = destination.replace('/', "--");
        let dest_path = models_dir.join(&dest_normalized);

        if dest_path.exists() {
            return Err(ModelError::InvalidConfig(format!(
                "Destination already exists: {}",
                dest_path.display()
            )));
        }

        println!(
            "Copying model from {} to {}...",
            source_path.display(),
            dest_path.display()
        );

        self.copy_dir_recursive(&source_path, &dest_path)?;

        let total_size = self.calculate_directory_size(&dest_path)?;

        println!("✓ Model copied successfully");
        println!("  Source: {}", source_path.display());
        println!("  Destination: {}", dest_path.display());
        println!("  Size: {}", Self::format_size(total_size));

        // Refresh cached model index (best-effort).
        if let Some(parent) = dest_path.parent() {
            let _ = crate::models::refresh_models_index(Some(parent));
        }

        Ok(())
    }

    pub fn info(&self, model: &str) -> Result<()> {
        let model_path = self.resolve_model_path(model)?;

        self.formatter
            .print_header(&format!("Model Information: {}", model));
        println!();

        let mut metadata = String::new();
        metadata.push_str(&format!("**Location:** `{}`\n\n", model_path.display()));

        let config_path = model_path.join("config.json");
        if config_path.exists() {
            let config_content = fs::read_to_string(&config_path)?;
            let config: Value = serde_json::from_str(&config_content)?;

            if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                metadata.push_str(&format!("**Type:** {}\n\n", model_type));
            }

            metadata.push_str("**Capabilities:**\n");

            let is_decoder = config
                .get("is_decoder")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let is_encoder_decoder = config
                .get("is_encoder_decoder")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_encoder_decoder {
                metadata.push_str("- Encoder-Decoder (Translation, Summarization)\n");
            } else if is_decoder {
                metadata.push_str("- Text Generation\n");
            } else {
                metadata.push_str("- Embeddings/Encoding\n");
            }

            if config.get("chat_template").is_some() {
                metadata.push_str("- Chat/Instruction Following\n");
            }

            metadata.push_str("\n");
        }

        let tokenizer_path = model_path.join("tokenizer.json");
        if tokenizer_path.exists() {
            metadata.push_str("**Tokenizer:** ✓ Present\n\n");
        }

        let weight_files = self.find_weight_files(&model_path)?;
        if !weight_files.is_empty() {
            metadata.push_str(&format!(
                "**Weight Files:** {} file(s)\n",
                weight_files.len()
            ));

            let has_safetensors = weight_files
                .iter()
                .any(|p| p.extension().map_or(false, |e| e == "safetensors"));
            let has_bin = weight_files
                .iter()
                .any(|p| p.extension().map_or(false, |e| e == "bin"));
            let has_gguf = weight_files
                .iter()
                .any(|p| p.extension().map_or(false, |e| e == "gguf"));

            if has_safetensors {
                metadata.push_str("- Format: SafeTensors ✓\n");
            }
            if has_bin {
                metadata.push_str("- Format: PyTorch (.bin)\n");
            }
            if has_gguf {
                metadata.push_str("- Format: GGUF (Quantized)\n");
            }
            metadata.push_str("\n");
        }

        let total_size = self.calculate_directory_size(&model_path)?;
        metadata.push_str(&format!(
            "**Disk Usage:** {}\n\n",
            Self::format_size(total_size)
        ));

        self.formatter.print_markdown(&metadata);

        Ok(())
    }

    pub fn verify(&self, model: &str) -> Result<()> {
        let model_path = self.resolve_model_path(model)?;

        self.formatter
            .print_header(&format!("Verifying Model: {}", model));
        println!();

        println!("Checking model integrity...\n");

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            println!("✗ config.json not found");
            return Err(ModelError::LocalModelError(
                "config.json missing".to_string(),
            ));
        }
        println!("✓ config.json found");

        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
        let mut tokenizer_found = false;
        for file in &tokenizer_files {
            if model_path.join(file).exists() {
                println!("✓ {} found", file);
                tokenizer_found = true;
                break;
            }
        }
        if !tokenizer_found {
            println!("✗ No tokenizer file found");
        }

        let weight_files = self.find_weight_files(&model_path)?;
        if weight_files.is_empty() {
            println!("✗ No weight files found");
            return Err(ModelError::LocalModelError(
                "No weight files found".to_string(),
            ));
        }

        println!("✓ Found {} weight file(s)", weight_files.len());

        println!("\nComputing checksums...\n");
        for file in &weight_files {
            let checksum = self.compute_file_checksum(file)?;
            println!(
                "  {} - SHA256: {}",
                file.file_name().unwrap().to_string_lossy(),
                &checksum[..16]
            );
        }

        let config_content = fs::read_to_string(&config_path)?;
        if serde_json::from_str::<Value>(&config_content).is_err() {
            println!("\n✗ config.json is invalid JSON");
            return Err(ModelError::LocalModelError(
                "Invalid config.json".to_string(),
            ));
        }
        println!("\n✓ config.json is valid JSON");

        println!("\n✓ Model verification complete");
        println!("  All essential files are present and accessible");

        Ok(())
    }

    fn calculate_directory_size(&self, path: &Path) -> Result<u64> {
        let mut total = 0u64;

        if path.is_file() {
            return Ok(fs::metadata(path)?.len());
        }

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total += metadata.len();
            } else if metadata.is_dir() {
                total += self.calculate_directory_size(&entry.path())?;
            }
        }

        Ok(total)
    }

    fn count_files(&self, path: &Path) -> Result<usize> {
        let mut count = 0;

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                count += 1;
            } else if metadata.is_dir() {
                count += self.count_files(&entry.path())?;
            }
        }

        Ok(count)
    }

    fn find_weight_files(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        let extensions = ["safetensors", "bin", "gguf", "pth", "pt"];

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if extensions.contains(&ext.to_string_lossy().as_ref()) {
                        files.push(path);
                    }
                }
            }
        }

        files.sort();
        Ok(files)
    }

    fn format_size(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", bytes)
        }
    }

    fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        fs::create_dir_all(dst)?;

        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if file_type.is_dir() {
                self.copy_dir_recursive(&src_path, &dst_path)?;
            } else {
                fs::copy(&src_path, &dst_path)?;
            }
        }

        Ok(())
    }

    fn compute_file_checksum(&self, path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
    }
}

impl Default for ModelOperations {
    fn default() -> Self {
        Self::new()
    }
}
