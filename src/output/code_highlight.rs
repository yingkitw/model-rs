use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

pub struct CodeHighlighter<'a> {
    syntax_set: &'a SyntaxSet,
    highlighter: HighlightLines<'a>,
}

impl<'a> CodeHighlighter<'a> {
    pub fn new(
        syntax_set: &'a SyntaxSet,
        syntax: &'a SyntaxReference,
        theme: &'a syntect::highlighting::Theme,
    ) -> Self {
        Self {
            syntax_set,
            highlighter: HighlightLines::new(syntax, theme),
        }
    }

    pub fn write(&mut self, code: &str) {
        for line in LinesWithEndings::from(code) {
            let ranges: Vec<(Style, &str)> = self
                .highlighter
                .highlight_line(line, self.syntax_set)
                .unwrap_or_default();
            let escaped = as_24_bit_terminal_escaped(&ranges[..], true);
            print!("{}", escaped);
        }
        print!("\x1b[0m");
    }

    pub fn finish_line(&mut self) {
        println!();
    }
}

pub fn resolve_syntax<'a>(syntax_set: &'a SyntaxSet, language: &str) -> &'a SyntaxReference {
    let lang = language.trim();
    if lang.is_empty() {
        return syntax_set.find_syntax_plain_text();
    }

    let lang_lc = lang.to_ascii_lowercase();
    let ext = match lang_lc.as_str() {
        "py" | "python" => "py",
        "rs" | "rust" => "rs",
        "js" | "javascript" => "js",
        "ts" | "typescript" => "ts",
        "sh" | "bash" | "shell" => "sh",
        "yml" | "yaml" => "yml",
        "md" | "markdown" => "md",
        "json" => "json",
        "toml" => "toml",
        _ => lang_lc.as_str(),
    };

    syntax_set
        .find_syntax_by_extension(ext)
        .or_else(|| syntax_set.find_syntax_by_extension(&lang_lc))
        .or_else(|| {
            syntax_set
                .syntaxes()
                .iter()
                .find(|s| s.name.to_ascii_lowercase() == lang_lc)
        })
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text())
}

pub fn resolve_theme(theme_set: &ThemeSet) -> &syntect::highlighting::Theme {
    let theme_name = if theme_set.themes.contains_key("Dark+ (default dark)") {
        "Dark+ (default dark)"
    } else if theme_set.themes.contains_key("Monokai Extended") {
        "Monokai Extended"
    } else if theme_set.themes.contains_key("base16-ocean.dark") {
        "base16-ocean.dark"
    } else {
        theme_set
            .themes
            .keys()
            .next()
            .map(|s| s.as_str())
            .unwrap_or("base16-ocean.dark")
    };

    theme_set
        .themes
        .get(theme_name)
        .unwrap_or_else(|| &theme_set.themes["base16-ocean.dark"])
}
