# train_transformer.py

import torch
import torch.nn as nn
import torch.optim as optim

from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM

# --- Daten laden ---
with open("data/google_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

vocab_size = tokenizer.vocab_size
print("Vokabular-Größe:", vocab_size)
print("Text-Länge:", len(encoded))

# --- Hyperparameter ---
block_size = 64          # Kontextlänge
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyTransformerLM(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=256,
    num_heads=4,
    num_layers=2,
    ff_dim=512,
    dropout=0.1,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()


def get_batch():
    # Sicherstellen, dass wir genug Länge haben
    max_start = len(encoded) - block_size - 1
    if max_start <= 0:
        raise ValueError("Text ist zu kurz für diesen block_size.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([encoded[i : i + block_size] for i in starts])
    y = torch.stack([encoded[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)


# --- Training Loop ---
num_steps = 3000

for step in range(num_steps):
    model.train()
    x, y = get_batch()

    optimizer.zero_grad()
    logits = model(x)

    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# --- Modell speichern ---
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "block_size": block_size,
    },
    "tiny_transformer.pt",
)

print("Transformer-Modell gespeichert als tiny_transformer.pt")