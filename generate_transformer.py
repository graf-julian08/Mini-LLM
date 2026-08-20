# generate_transformer.py

import torch
import torch.nn.functional as F

from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Daten & Tokenizer laden ---
with open("data/dialogs.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = CharTokenizer(raw_text)
vocab_size = tokenizer.vocab_size

# --- Modell laden ---
checkpoint = torch.load("tiny_transformer.pt", map_location=device)
block_size = checkpoint["block_size"]

model = TinyTransformerLM(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=256,
    num_heads=4,
    num_layers=2,
    ff_dim=512,
    dropout=0.1,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def generate_ids(model, idx, max_new_tokens: int, block_size: int, device):
    """
    Autoregressive Generierung.
    idx: (1, T) Start-IDs
    """
    for _ in range(max_new_tokens):
        # Kontext kürzen auf block_size
        idx_cond = idx[:, -block_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # (1, T_cond, Vocab)

        # Letztes Token
        logits_last = logits[:, -1, :]  # (1, Vocab)
        probs = F.softmax(logits_last, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx


def generate_text(prompt: str, max_new_tokens: int = 200):
    """
    Prompt-Text -> generierter Text.
    """
    encoded = tokenizer.encode(prompt)
    if len(encoded) == 0:
        encoded = [0]

    x = torch.tensor([encoded], dtype=torch.long, device=device)

    out_ids = generate_ids(model, x, max_new_tokens=max_new_tokens, block_size=block_size, device=device)
    out_ids = out_ids[0].tolist()
    return tokenizer.decode(out_ids)


if __name__ == "__main__":
    test_prompt = "User: Hallo\nBot:"
    out = generate_text(test_prompt, max_new_tokens=200)
    print(out)