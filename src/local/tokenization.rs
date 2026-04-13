use crate::error::Result;
use tokenizers::Tokenizer;

pub fn get_eos_token(tokenizer: &Tokenizer) -> Option<u32> {
    tokenizer.token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<EOS>"))
}

pub fn stream_piece(tokenizer: &Tokenizer, token_id: u32, started: &mut bool) -> Result<Option<String>> {
    if let Some(raw) = tokenizer.id_to_token(token_id) {
        if raw == "</s>" || raw == "<s>" || raw == "<unk>" {
            return Ok(None);
        }

        let text = tokenizer.decode(&[token_id], false)?;
        if text.is_empty() {
            return Ok(None);
        }

        let mut piece = String::new();
        if raw.starts_with('▁') && *started {
            piece.push(' ');
        }
        piece.push_str(&text);
        *started = true;
        return Ok(Some(piece));
    }

    Ok(None)
}
