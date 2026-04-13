use std::io::{self, Write};

/// ANSI color codes for terminal output
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const ITALIC: &str = "\x1b[3m";
    pub const UNDERLINE: &str = "\x1b[4m";

    // Foreground colors
    pub const BLACK: &str = "\x1b[30m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";

    // Bright foreground colors
    pub const BRIGHT_BLACK: &str = "\x1b[90m";
    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const BRIGHT_WHITE: &str = "\x1b[97m";

    // Background colors
    pub const BG_BLACK: &str = "\x1b[40m";
    pub const BG_RED: &str = "\x1b[41m";
    pub const BG_GREEN: &str = "\x1b[42m";
    pub const BG_YELLOW: &str = "\x1b[43m";
    pub const BG_BLUE: &str = "\x1b[44m";
    pub const BG_MAGENTA: &str = "\x1b[45m";
    pub const BG_CYAN: &str = "\x1b[46m";
    pub const BG_WHITE: &str = "\x1b[47m";
}

/// Simple Markdown renderer for terminal output
pub struct MarkdownRenderer {
    in_code_block: bool,
    code_language: Option<String>,
    in_inline_code: bool,
}

impl MarkdownRenderer {
    pub fn new() -> Self {
        Self {
            in_code_block: false,
            code_language: None,
            in_inline_code: false,
        }
    }

    /// Render markdown text to terminal-formatted output
    pub fn render(&mut self, text: &str) -> String {
        let mut result = String::new();
        let mut lines = text.lines().peekable();

        while let Some(line) = lines.next() {
            // Check for code block fence
            if line.starts_with("```") {
                if self.in_code_block {
                    // Closing code block
                    result.push_str(&format!("{}{}\n", colors::DIM, "─".repeat(80)));
                    self.in_code_block = false;
                    self.code_language = None;
                } else {
                    // Opening code block
                    let lang = line[3..].trim().to_string();
                    self.code_language = if lang.is_empty() { None } else { Some(lang) };
                    result.push_str(&format!("{}Code Block{}\n", colors::CYAN, colors::RESET));
                    if let Some(ref lang) = self.code_language {
                        result.push_str(&format!("{}Language: {}{}\n", colors::DIM, lang, colors::RESET));
                    }
                    result.push_str(&format!("{}\n", colors::BRIGHT_BLACK));
                    self.in_code_block = true;
                }
                continue;
            }

            // Handle content inside code blocks
            if self.in_code_block {
                result.push_str(line);
                result.push('\n');
                if lines.peek().is_none() {
                    result.push_str(colors::RESET);
                }
                continue;
            }

            // Process inline markdown
            let processed = self.process_inline_markdown(line);
            result.push_str(&processed);
            result.push('\n');
        }

        result
    }

    fn process_inline_markdown(&mut self, line: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Check for inline code
            if chars[i] == '`' && !self.in_code_block {
                if self.in_inline_code {
                    result.push_str(colors::RESET);
                    self.in_inline_code = false;
                } else {
                    result.push_str(&format!("{}{}", colors::BG_BLACK, colors::BRIGHT_WHITE));
                    self.in_inline_code = true;
                }
                i += 1;
                continue;
            }

            // Check for bold
            if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
                if let Some(end) = find_closing_double_star(&chars, i + 2) {
                    result.push_str(&format!("{}{}", colors::BOLD, colors::BRIGHT_WHITE));
                    for j in (i + 2)..end {
                        result.push(chars[j]);
                    }
                    result.push_str(colors::RESET);
                    i = end + 2;
                    continue;
                }
            }

            // Check for italic
            if chars[i] == '*' && (i == 0 || chars[i - 1] != '*') {
                if let Some(end) = find_closing_single_star(&chars, i + 1) {
                    result.push_str(&format!("{}{}", colors::ITALIC, colors::WHITE));
                    for j in (i + 1)..end {
                        result.push(chars[j]);
                    }
                    result.push_str(colors::RESET);
                    i = end + 1;
                    continue;
                }
            }

            // Check for headers
            if chars[i] == '#' && (i == 0 || chars[i - 1] == ' ') {
                let mut count = 0;
                while i + count < chars.len() && chars[i + count] == '#' {
                    count += 1;
                }
                if i + count < chars.len() && chars[i + count] == ' ' {
                    let header_text: String = chars[(i + count + 1)..].iter().collect();
                    result.push_str(&format!("\n{}{}{}\n", colors::BOLD, colors::CYAN, "#".repeat(count)));
                    result.push_str(&format!("{}{}{}{}\n\n", colors::BOLD, colors::BRIGHT_CYAN, header_text.trim(), colors::RESET));
                    return result;
                }
            }

            // Regular character
            result.push(chars[i]);
            i += 1;
        }

        // Reset inline code if still open at end of line
        if self.in_inline_code {
            result.push_str(colors::RESET);
            self.in_inline_code = false;
        }

        result
    }
}

impl Default for MarkdownRenderer {
    fn default() -> Self {
        Self::new()
    }
}

fn find_closing_double_star(chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len() - 1 {
        if chars[i] == '*' && chars[i + 1] == '*' {
            return Some(i);
        }
    }
    None
}

fn find_closing_single_star(chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len() {
        if chars[i] == '*' && (i + 1 >= chars.len() || chars[i + 1] != '*') {
            return Some(i);
        }
    }
    None
}

/// Print a styled header
pub fn print_header(text: &str) {
    println!("\n{}{}�═══════════════════════════════════════════════════════════════════{}", colors::BOLD, colors::CYAN, colors::RESET);
    println!("{}{}  {}{}", colors::BOLD, colors::CYAN, text, colors::RESET);
    println!("{}{}╠═══════════════════════════════════════════════════════════════════{}\n", colors::BOLD, colors::CYAN, colors::RESET);
}

/// Print a styled section header
pub fn print_section(text: &str) {
    println!("\n{}{}▸ {}{}", colors::BOLD, colors::BRIGHT_CYAN, text, colors::RESET);
}

/// Print a success message
pub fn print_success(text: &str) {
    println!("{}✓{} {}", colors::BRIGHT_GREEN, colors::RESET, text);
}

/// Print an error message
pub fn print_error(text: &str) {
    println!("{}✗{} {}{}{}", colors::BRIGHT_RED, colors::RESET, colors::BRIGHT_RED, text, colors::RESET);
}

/// Print a warning message
pub fn print_warning(text: &str) {
    println!("{}⚠{} {}{}{}", colors::BRIGHT_YELLOW, colors::RESET, colors::BRIGHT_YELLOW, text, colors::RESET);
}

/// Print an info message
pub fn print_info(text: &str) {
    println!("{}ℹ{} {}", colors::BRIGHT_BLUE, colors::RESET, text);
}

/// Print user message in chat
pub fn print_user_message(text: &str) {
    println!("\n{}{}You:{} {}", colors::BOLD, colors::BRIGHT_GREEN, colors::RESET, text);
}

/// Print assistant message in chat
pub fn print_assistant_header() {
    print!("{}{}Assistant:{} ", colors::BOLD, colors::BRIGHT_CYAN, colors::RESET);
    io::stdout().flush().unwrap();
}

/// Print system message in chat
pub fn print_system_message(text: &str) {
    println!("{}{}System:{} {}", colors::DIM, colors::BRIGHT_BLACK, colors::RESET, text);
}

/// Print a divider line
pub fn print_divider() {
    println!("{}{}{}{}", colors::DIM, "─".repeat(80), colors::RESET, colors::RESET);
}

/// Print a banner for chat mode
pub fn print_chat_banner() {
    println!("\n{}{}╭─────────────────────────────────────────────────────────────────╮{}", colors::BOLD, colors::CYAN, colors::RESET);
    println!("{}{}│{}  {}Interactive Chat Mode{}                                        {}│{}", colors::BOLD, colors::CYAN, colors::RESET, colors::BOLD, colors::BRIGHT_WHITE, colors::CYAN, colors::RESET);
    println!("{}{}│{}  {}Type your message and press Enter to send{}                    {}│{}", colors::BOLD, colors::CYAN, colors::RESET, colors::WHITE, colors::CYAN, colors::RESET, colors::RESET);
    println!("{}{}│{}  {}Type 'quit', 'exit', or Ctrl+C to exit{}                       {}│{}", colors::BOLD, colors::CYAN, colors::RESET, colors::WHITE, colors::CYAN, colors::RESET, colors::RESET);
    println!("{}{}╰─────────────────────────────────────────────────────────────────╯{}\n", colors::BOLD, colors::CYAN, colors::RESET);
}

/// Format and print generated text with markdown rendering
pub fn print_markdown(text: &str) {
    let mut renderer = MarkdownRenderer::new();
    let rendered = renderer.render(text);
    print!("{}", rendered);
    io::stdout().flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_renderer_basic() {
        let mut renderer = MarkdownRenderer::new();
        let input = "Hello **world**";
        let output = renderer.render(input);
        assert!(output.contains("Hello"));
        assert!(output.contains("world"));
    }

    #[test]
    fn test_markdown_renderer_code_block() {
        let mut renderer = MarkdownRenderer::new();
        let input = "```rust\nfn main() {}\n```";
        let output = renderer.render(input);
        assert!(output.contains("Code Block"));
        assert!(output.contains("Language: rust"));
    }

    #[test]
    fn test_markdown_renderer_inline_code() {
        let mut renderer = MarkdownRenderer::new();
        let input = "Use `cargo build` to compile";
        let output = renderer.render(input);
        assert!(output.contains("cargo build"));
    }

    #[test]
    fn test_markdown_renderer_headers() {
        let mut renderer = MarkdownRenderer::new();
        let input = "# Title\n\n## Subtitle";
        let output = renderer.render(input);
        assert!(output.contains("#"));
        assert!(output.contains("Title"));
        assert!(output.contains("Subtitle"));
    }
}
