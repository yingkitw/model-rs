#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Text,
    Lang,
    Code,
}

pub enum CodeStreamEvent<'a> {
    Start { language: &'a str },
    Chunk { language: &'a str, code: &'a str },
    End,
}

pub struct MarkdownStreamRenderer {
    state: StreamState,
    buffer: String,
    lang: String,
    code: String,
}

impl MarkdownStreamRenderer {
    pub fn new() -> Self {
        Self {
            state: StreamState::Text,
            buffer: String::new(),
            lang: String::new(),
            code: String::new(),
        }
    }

    pub fn push_with<FText, FCode>(&mut self, chunk: &str, mut on_text: FText, mut on_code: FCode)
    where
        FText: FnMut(&str),
        FCode: FnMut(CodeStreamEvent<'_>),
    {
        self.buffer.push_str(chunk);
        self.drain(&mut on_text, &mut on_code);
    }

    pub fn finish_with<FText, FCode>(&mut self, mut on_text: FText, mut on_code: FCode)
    where
        FText: FnMut(&str),
        FCode: FnMut(CodeStreamEvent<'_>),
    {
        self.drain(&mut on_text, &mut on_code);

        match self.state {
            StreamState::Text => {
                if !self.buffer.is_empty() {
                    on_text(&self.buffer);
                    self.buffer.clear();
                }
            }
            StreamState::Lang => {
                if !self.lang.is_empty() {
                    on_text(&self.lang);
                    self.lang.clear();
                }
                self.state = StreamState::Text;
            }
            StreamState::Code => {
                self.code.push_str(&self.buffer);
                self.buffer.clear();
                if !self.code.is_empty() {
                    let lang = self.lang.trim();
                    on_code(CodeStreamEvent::Start { language: lang });
                    on_code(CodeStreamEvent::Chunk {
                        language: lang,
                        code: &self.code,
                    });
                    on_code(CodeStreamEvent::End);
                }
                self.lang.clear();
                self.code.clear();
                self.state = StreamState::Text;
            }
        }
    }

    fn drain<FText, FCode>(&mut self, on_text: &mut FText, on_code: &mut FCode)
    where
        FText: FnMut(&str),
        FCode: FnMut(CodeStreamEvent<'_>),
    {
        loop {
            match self.state {
                StreamState::Text => {
                    if let Some(pos) = self.buffer.find("```") {
                        if pos > 0 {
                            on_text(&self.buffer[..pos]);
                        }
                        self.buffer.drain(..pos + 3);
                        self.lang.clear();
                        self.code.clear();
                        self.state = StreamState::Lang;
                        continue;
                    }

                    let keep = self.trailing_backticks_to_keep();
                    let emit_len = self.buffer.len().saturating_sub(keep);
                    if emit_len > 0 {
                        on_text(&self.buffer[..emit_len]);
                        self.buffer.drain(..emit_len);
                    }
                    break;
                }
                StreamState::Lang => {
                    if let Some(nl) = self.buffer.find('\n') {
                        self.lang.push_str(&self.buffer[..nl]);
                        self.buffer.drain(..nl + 1);
                        let lang = self.lang.trim();
                        on_code(CodeStreamEvent::Start { language: lang });
                        self.state = StreamState::Code;
                        continue;
                    }

                    if !self.buffer.is_empty() {
                        self.lang.push_str(&self.buffer);
                        self.buffer.clear();
                    }
                    break;
                }
                StreamState::Code => {
                    if let Some(pos) = self.buffer.find("```") {
                        if pos > 0 {
                            self.code.push_str(&self.buffer[..pos]);
                        }
                        self.buffer.drain(..pos + 3);
                        if self.buffer.starts_with('\n') {
                            self.buffer.drain(..1);
                        }
                        let lang = self.lang.trim();
                        while let Some(nl) = self.code.find('\n') {
                            let line = &self.code[..nl + 1];
                            on_code(CodeStreamEvent::Chunk {
                                language: lang,
                                code: line,
                            });
                            self.code.drain(..nl + 1);
                        }
                        if !self.code.is_empty() {
                            let rest = std::mem::take(&mut self.code);
                            on_code(CodeStreamEvent::Chunk {
                                language: lang,
                                code: &rest,
                            });
                        }
                        on_code(CodeStreamEvent::End);
                        self.lang.clear();
                        self.code.clear();
                        self.state = StreamState::Text;
                        continue;
                    }

                    let keep = self.trailing_backticks_to_keep();
                    let emit_len = self.buffer.len().saturating_sub(keep);
                    if emit_len > 0 {
                        self.code.push_str(&self.buffer[..emit_len]);
                        self.buffer.drain(..emit_len);

                        let lang = self.lang.trim();
                        while let Some(nl) = self.code.find('\n') {
                            let line = &self.code[..nl + 1];
                            on_code(CodeStreamEvent::Chunk {
                                language: lang,
                                code: line,
                            });
                            self.code.drain(..nl + 1);
                        }
                    }
                    break;
                }
            }
        }
    }

    fn trailing_backticks_to_keep(&self) -> usize {
        let bytes = self.buffer.as_bytes();
        let mut count = 0usize;
        let mut i = bytes.len();
        while count < 2 && i > 0 {
            if bytes[i - 1] == b'`' {
                count += 1;
                i -= 1;
            } else {
                break;
            }
        }
        count
    }
}

impl Default for MarkdownStreamRenderer {
    fn default() -> Self {
        Self::new()
    }
}
