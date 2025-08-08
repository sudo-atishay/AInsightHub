from typing import Dict, List, Tuple

from tiktoken import get_encoding

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# Wrapper class to make OpenAI's tokenizer compatible with HybridChunker

class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    """Minimal Wrapper for OpenAI's tokenizer."""

    def __init__(self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs):
        """Initialize the tokenizer.

        Args:
            model_name: The name of the OpenAI encoding to use
            max_length: Maximum sequence length
        """
        super().__init__(model_max_length=max_length, **kwargs)
        self.encoding = get_encoding(model_name)
        
        
    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> Dict[str, int]:
        return dict(enumerate(range(self.vocab_size)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, *args) -> Tuple[str]:
        return ()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Class method to match HuggingFace's interface."""
        return cls()