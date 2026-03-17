from __future__ import annotations

from .base import BaseMTModel


class Seq2SeqModel(BaseMTModel):
    """Encoder-decoder translation model (e.g. NLLB-200).

    Only source text and language codes.
    Question and answer are translated separately.

    example language codes for NLLB-200:
       -  English  →   "eng_Latn"
       -  Twi      →   "twi_Latn"
       -  Igbo     →   "ibo_Latn"
       -  Hausa    →   "hau_Latn"
    """

    MODEL_TYPE = 'seq2seq'

    def __init__(
        self,
        model_name: str,
        src_lang: str,
        tgt_lang: str,
        max_tokens: int = 512,
        device: int | None = None,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.pipe = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=max_tokens,
            device=device,
        )

    def translate_text(self, text: str) -> str:
        return self.pipe(text)[0]['translation_text']

    def translate(self, messages: list[dict]) -> str:
        # Fallback: extract user content and translate as a block.
        # Prefer translate_text() called directly from run.py for seq2seq models.
        user_text = next(m['content'] for m in messages if m['role'] == 'user')
        return self.translate_text(user_text)
