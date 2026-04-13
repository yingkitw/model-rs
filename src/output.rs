use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use termimad::{crossterm::style::Color, MadSkin};

mod code_highlight;
mod markdown_stream;

pub use code_highlight::CodeHighlighter;
pub use markdown_stream::{CodeStreamEvent, MarkdownStreamRenderer};

pub struct OutputFormatter {
    skin: MadSkin,
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl OutputFormatter {
    pub fn new() -> Self {
        let mut skin = MadSkin::default();

        skin.bold.set_fg(Color::Cyan);
        skin.italic.set_fg(Color::Yellow);
        skin.headers[0].set_fg(Color::Green);
        skin.headers[1].set_fg(Color::Blue);
        skin.headers[2].set_fg(Color::Magenta);
        skin.code_block.set_fg(Color::White);
        skin.inline_code.set_fg(Color::Yellow);

        let syntax_set = SyntaxSet::load_defaults_newlines();
        let theme_set = ThemeSet::load_defaults();

        Self {
            skin,
            syntax_set,
            theme_set,
        }
    }

    pub fn print_markdown_fragment(&self, text: &str) {
        let looks_like_markdown = text.contains("**")
            || text.contains("__")
            || text.contains('`')
            || text.starts_with('#')
            || text.starts_with("- ")
            || text.contains("\n#")
            || text.contains("\n- ");

        if looks_like_markdown {
            self.skin.print_text(text);
        } else {
            print!("{}", text);
        }
    }

    pub fn print_markdown(&self, text: &str) {
        // Parse and handle code blocks with syntax highlighting
        let mut current_pos = 0;
        let text_bytes = text.as_bytes();

        while current_pos < text.len() {
            // Look for code block start
            if let Some(code_start) = text[current_pos..].find("```") {
                let absolute_start = current_pos + code_start;

                // Print text before code block
                if code_start > 0 {
                    let before_text = &text[current_pos..absolute_start];
                    self.skin.print_text(before_text);
                }

                // Find language identifier (first line after ```)
                let after_marker = absolute_start + 3;
                let line_end = text[after_marker..].find('\n').unwrap_or(0);
                let lang_line = &text[after_marker..after_marker + line_end].trim();

                // Find code block end
                if let Some(code_end_offset) = text[after_marker + line_end..].find("```") {
                    let code_end = after_marker + line_end + code_end_offset;
                    let code_content = &text[after_marker + line_end + 1..code_end];

                    // Print code block with syntax highlighting
                    println!();
                    self.print_code(code_content, lang_line);

                    // Move past the closing ```
                    current_pos = code_end + 3;

                    // Skip newline after closing ```
                    if current_pos < text.len() && text_bytes[current_pos] == b'\n' {
                        current_pos += 1;
                    }
                } else {
                    // No closing ```, treat as regular text
                    self.skin.print_text(&text[current_pos..]);
                    break;
                }
            } else {
                // No more code blocks, print remaining text
                self.skin.print_text(&text[current_pos..]);
                break;
            }
        }
    }

    pub fn print_code(&self, code: &str, language: &str) {
        let mut h = self.code_highlighter(language);
        h.write(code);
        h.finish_line();
    }

    pub fn code_highlighter<'a>(&'a self, language: &str) -> CodeHighlighter<'a> {
        let syntax = code_highlight::resolve_syntax(&self.syntax_set, language);
        let theme = code_highlight::resolve_theme(&self.theme_set);
        CodeHighlighter::new(&self.syntax_set, syntax, theme)
    }

    pub fn print_header(&self, text: &str) {
        self.print_markdown(&format!("# {}\n", text));
    }

    pub fn print_section(&self, title: &str, content: &str) {
        self.print_markdown(&format!("## {}\n\n{}\n", title, content));
    }

    pub fn print_info(&self, text: &str) {
        self.print_markdown(&format!("**ℹ️  {}**\n", text));
    }

    pub fn print_success(&self, text: &str) {
        self.print_markdown(&format!("**✓ {}**\n", text));
    }

    pub fn print_warning(&self, text: &str) {
        self.print_markdown(&format!("**⚠️  {}**\n", text));
    }

    pub fn print_error(&self, text: &str) {
        self.print_markdown(&format!("**❌ {}**\n", text));
    }

    pub fn print_list_item(&self, text: &str) {
        self.print_markdown(&format!("- {}\n", text));
    }

    pub fn print_model_info(
        &self,
        name: &str,
        path: &str,
        format: &str,
        arch: &str,
        size: &str,
        files: usize,
    ) {
        let info = format!(
            r#"### {}

- **Path:** `{}`
- **Format:** {}
- **Architecture:** {}
- **Size:** {} ({} files)
"#,
            name, path, format, arch, size, files
        );
        self.print_markdown(&info);
    }

    pub fn print_search_result(
        &self,
        idx: usize,
        model_id: &str,
        author: Option<&str>,
        task: Option<&str>,
        downloads: Option<u64>,
        likes: Option<u64>,
        library: Option<&str>,
    ) {
        let mut result = format!("### {}. {}\n\n", idx, model_id);

        if let Some(a) = author {
            result.push_str(&format!("- **Author:** {}\n", a));
        }
        if let Some(t) = task {
            result.push_str(&format!("- **Task:** {}\n", t));
        }
        if let Some(d) = downloads {
            result.push_str(&format!("- **Downloads:** {}\n", d));
        }
        if let Some(l) = likes {
            result.push_str(&format!("- **Likes:** {}\n", l));
        }
        if let Some(lib) = library {
            result.push_str(&format!("- **Library:** {}\n", lib));
        }

        result.push_str(&format!(
            "\n**Download:** `model-rs download {}`\n",
            model_id
        ));

        self.print_markdown(&result);
    }

    pub fn print_chat_header(&self) {
        let header = r#"
# Interactive Chat Mode

**Commands:**
- Type your messages and press Enter
- `/help` - Show available commands
- `/quit` or `/exit` - Exit chat mode
"#;
        self.print_markdown(header);
    }

    pub fn print_help_commands(&self) {
        let help = r#"
## Chat Commands

- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/save <filename>` - Save conversation to file
- `/load <filename>` - Load conversation from file
- `/set <param> <val>` - Change parameters (temperature, top_p)
- `/search <query>` - Search conversation history
- `/quit` or `/exit` - Exit chat mode

**Example:** `/set temperature 0.8`
"#;
        self.print_markdown(help);
    }
}

impl Default for OutputFormatter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_formatter_creation() {
        let formatter = OutputFormatter::new();
        assert!(formatter.syntax_set.syntaxes().len() > 0);
        assert!(formatter.theme_set.themes.len() > 0);
    }

    #[test]
    fn test_markdown_stream_renderer_plain_text() {
        let mut r = MarkdownStreamRenderer::new();
        let out = std::cell::RefCell::new(String::new());
        let codes = std::cell::RefCell::new(Vec::<(String, String)>::new());
        let cur_lang = std::cell::RefCell::new(String::new());
        let cur_code = std::cell::RefCell::new(String::new());

        r.push_with(
            "hello\nworld\n",
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );
        r.finish_with(
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );

        assert_eq!(out.borrow().as_str(), "hello\nworld\n");
        assert!(codes.borrow().is_empty());
    }

    #[test]
    fn test_markdown_stream_renderer_code_block_split_fence() {
        let mut r = MarkdownStreamRenderer::new();
        let out = std::cell::RefCell::new(String::new());
        let codes = std::cell::RefCell::new(Vec::<(String, String)>::new());
        let cur_lang = std::cell::RefCell::new(String::new());
        let cur_code = std::cell::RefCell::new(String::new());

        r.push_with(
            "before\n``",
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );
        r.push_with(
            "`rust\nfn main() {}\n``",
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );
        r.push_with(
            "`\nafter\n",
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );
        r.finish_with(
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );

        assert_eq!(out.borrow().as_str(), "before\nafter\n");
        let codes = codes.borrow();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].0, "rust");
        assert_eq!(codes[0].1, "fn main() {}\n");
    }

    #[test]
    fn test_markdown_stream_renderer_no_lang() {
        let mut r = MarkdownStreamRenderer::new();
        let out = std::cell::RefCell::new(String::new());
        let codes = std::cell::RefCell::new(Vec::<(String, String)>::new());
        let cur_lang = std::cell::RefCell::new(String::new());
        let cur_code = std::cell::RefCell::new(String::new());

        r.push_with(
            "```\nline1\nline2\n```\n",
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );
        r.finish_with(
            |t| out.borrow_mut().push_str(t),
            |ev| match ev {
                CodeStreamEvent::Start { language } => {
                    cur_lang.borrow_mut().clear();
                    cur_lang.borrow_mut().push_str(language);
                    cur_code.borrow_mut().clear();
                }
                CodeStreamEvent::Chunk { language: _, code } => {
                    cur_code.borrow_mut().push_str(code);
                }
                CodeStreamEvent::End => {
                    codes
                        .borrow_mut()
                        .push((cur_lang.borrow().to_string(), cur_code.borrow().to_string()));
                    cur_lang.borrow_mut().clear();
                    cur_code.borrow_mut().clear();
                }
            },
        );

        assert_eq!(out.borrow().as_str(), "");
        let codes = codes.borrow();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].0, "");
        assert_eq!(codes[0].1, "line1\nline2\n");
    }
}
