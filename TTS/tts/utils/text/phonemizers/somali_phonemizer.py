from TTS.tts.utils.text.somali.phonemizer import somali_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from typing import Optional
import re


class SO_SO_Phonemizer(BasePhonemizer):
    language = "so-so"

    def __init__(self, punctuations: str = ".,:;?!", keep_puncs: bool = True, **kwargs):
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "so_so_phonemizer"

    def _phonemize(self, text: str, separator: Optional[str] = "|") -> str:
        ph = somali_text_to_phonemes(text)
        ph = re.sub(r"\s+", " ", ph).strip()
        if separator is None or separator == "":
            return ph.replace(" ", "")
        return separator.join(ph.split(" "))

    def phonemize(self, text: str, separator="|", language=None) -> str:
        return self._phonemize(text, separator)

    @staticmethod
    def supported_languages() -> dict:
        return {"so-so": "Somali (Somalia)"}

    def version(self) -> str:
        return "0.0.2"

    def is_available(self) -> bool:
        return True

# --- Quick test ---
if __name__ == "__main__":
    print("faa'ido →", somali_text_to_phonemes("faa'ido"))
    print("dhagax →", somali_text_to_phonemes("dhagax"))
    print("qab →", somali_text_to_phonemes("qab"))
    print("shaqayn →", somali_text_to_phonemes("shaqayn"))
