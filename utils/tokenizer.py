# utils/tokenizer.py

class CharTokenizer:
    """
    Simpler Character-Tokenizer.
    Wandelt Text in IDs und wieder zurück.
    """

    def __init__(self, text: str):
        # Alle Zeichen, die im Trainings-Text vorkommen
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Mapping char <-> index
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str):
        """
        Text -> Liste von IDs.
        Unbekannte Zeichen gehen auf ID 0.
        """
        return [self.char2idx.get(ch, 0) for ch in text]

    def decode(self, ids):
        """
        Liste von IDs -> Text.
        Unbekannte IDs werden zu '?'.
        """
        return "".join(self.idx2char.get(i, "?") for i in ids)