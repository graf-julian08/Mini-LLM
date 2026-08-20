#!/usr/bin/env python3
"""
Mini LLM Auto-Trainer: Trainiert sich selbst auf verschiedenen Themen
Sammelt Daten, trainiert kontinuierlich und speichert den Fortschritt
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime

from utils.google_search import google_search
from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM


# ===== KONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FILE = "data/combined_corpus.txt"
MODEL_FILE = "tiny_transformer.pt"
TRAINING_LOG = "training_log.json"

# Hyperparameter
BLOCK_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 512
DROPOUT = 0.1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
BATCHES_PER_EPOCH = 50

# Training Topics (verschiedene Themen zum Lernen)
TRAINING_TOPICS = [
    "künstliche intelligenz",
    "schwarze löcher",
    "covid-19",
    "klimawandel",
    "quantencomputer",
    "neurologie gehirn",
    "biotechnologie crispr",
    "blockchain kryptowährung",
    "raumfahrt raumstation",
    "energie solarenergie",
]


# ===== STEP 1: MULTI-TOPIC DATEN SAMMELN =====
def collect_multi_topic_data() -> str:
    """
    Sammelt Google-Daten zu mehreren Themen parallel.
    Baut einen großen Corpus aus verschiedenen Wissensgebieten auf.
    """
    print("\n" + "="*60)
    print("📡 STEP 1: MULTI-TOPIC DATEN SAMMELN")
    print("="*60)
    
    all_texts = []
    total_results = 0
    
    for topic_idx, topic in enumerate(TRAINING_TOPICS, 1):
        print(f"\n  [{topic_idx}/{len(TRAINING_TOPICS)}] Thema: '{topic}'")
        
        # Verschiedene Suchanfragen pro Thema
        queries = [
            topic,
            f"erklärung {topic}",
            f"was ist {topic}",
            f"info {topic}",
            f"{topic} einfach erklärt",
        ]
        
        for q in queries:
            try:
                results = google_search(q, num_results=3)
                
                for r in results:
                    title = r["title"]
                    snippet = r["snippet"]
                    clean_text = f"{title}. {snippet}".replace("\n", " ")
                    all_texts.append(clean_text)
                    total_results += 1
            except Exception as e:
                continue
        
        print(f"     ✓ {total_results} Texte gesammelt")
    
    # Corpus speichern
    collected_text = "\n".join(all_texts)
    os.makedirs("data", exist_ok=True)
    
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(collected_text)
    
    print(f"\n  ✅ {total_results} Texte zu {len(TRAINING_TOPICS)} Themen gesammelt!")
    print(f"  📝 Corpus: {len(collected_text):,} Zeichen")
    
    return collected_text


# ===== STEP 2: MODELL INITIALISIEREN ODER LADEN =====
def setup_model(training_data: str, load_existing: bool = True) -> tuple:
    """
    Erstellt oder lädt das Modell.
    """
    print("\n" + "="*60)
    print("🤖 STEP 2: MODELL SETUP")
    print("="*60)
    
    tokenizer = CharTokenizer(training_data)
    vocab_size = tokenizer.vocab_size
    
    print(f"\n  📊 Tokenizer:")
    print(f"     • Vokabular: {vocab_size}")
    print(f"     • Text-Länge: {len(training_data):,} Zeichen")
    
    # Model laden oder neu erstellen
    model_loaded = False
    if load_existing and os.path.exists(MODEL_FILE):
        print(f"\n  📂 Versuche existierendes Modell zu laden...")
        try:
            checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
            saved_vocab = checkpoint.get("vocab_size", 0)
            
            # Nur laden wenn Vokabular passt
            if saved_vocab == vocab_size:
                model = TinyTransformerLM(
                    vocab_size=vocab_size,
                    block_size=BLOCK_SIZE,
                    embed_dim=EMBED_DIM,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    ff_dim=FF_DIM,
                    dropout=DROPOUT,
                ).to(DEVICE)
                
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"     ✓ Modell geladen (trainiert: {checkpoint.get('epochs_trained', 0)} Epochen)")
                model_loaded = True
            else:
                print(f"     ⚠️  Vokabular-Mismatch: {saved_vocab} → {vocab_size}")
                print(f"     → Erstelle neues Modell...")
        except Exception as e:
            print(f"     ⚠️  Fehler beim Laden: {e}")
            print(f"     → Erstelle neues Modell...")
    
    if not model_loaded:
        print(f"\n  🏗️  Erstelle neues Modell...")
        model = TinyTransformerLM(
            vocab_size=vocab_size,
            block_size=BLOCK_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
        ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  🏗️  Modell-Info:")
    print(f"     • Parameter: {total_params:,}")
    print(f"     • Device: {DEVICE}")
    
    return model, tokenizer


# ===== STEP 3: KONTINUIERLICHES TRAINING =====
def continuous_training(model: TinyTransformerLM, tokenizer: CharTokenizer, training_data: str, num_rounds: int = 3):
    """
    Trainiert das Modell über mehrere Runden.
    Speichert Fortschritt nach jeder Runde.
    """
    print("\n" + "="*60)
    print("⚙️  STEP 3: KONTINUIERLICHES TRAINING")
    print("="*60)
    
    # Training-Log laden
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, "r") as f:
            log = json.load(f)
    else:
        log = {
            "total_epochs": 0,
            "history": []
        }
    
    vocab_size = tokenizer.vocab_size
    encoded = torch.tensor(tokenizer.encode(training_data), dtype=torch.long)
    max_start = len(encoded) - BLOCK_SIZE - 1
    
    if max_start <= 0:
        print(f"  ⚠️  Text zu kurz! max_start = {max_start}")
        max_start = max(0, max_start)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
    for round_num in range(num_rounds):
        print(f"\n  🔄 Trainings-Runde {round_num + 1}/{num_rounds}")
        print(f"  {'-'*56}")
        
        round_losses = []
        
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            num_batches = 0
            
            for step in range(BATCHES_PER_EPOCH):
                try:
                    starts = torch.randint(0, max(max_start, 1), (BATCH_SIZE,))
                    x = torch.stack([encoded[i : i + BLOCK_SIZE] for i in starts])
                    y = torch.stack([encoded[i + 1 : i + 1 + BLOCK_SIZE] for i in starts])
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            round_losses.append(avg_loss)
            
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"     • Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_loss:.4f}")
        
        # Log speichern
        log["total_epochs"] += NUM_EPOCHS
        log["history"].append({
            "round": round_num + 1,
            "timestamp": datetime.now().isoformat(),
            "avg_loss": sum(round_losses) / len(round_losses),
            "epochs": NUM_EPOCHS
        })
        
        # Modell speichern
        print(f"\n     💾 Speichere Modell...")
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": BLOCK_SIZE,
            "epochs_trained": log["total_epochs"],
        }, MODEL_FILE)
        
        # Log speichern
        with open(TRAINING_LOG, "w") as f:
            json.dump(log, f, indent=2)
        
        print(f"     ✓ Modell gespeichert")
    
    model.eval()
    return model, log


# ===== STEP 4: TEST-GENERIERUNG =====
def test_generation(model: TinyTransformerLM, tokenizer: CharTokenizer, num_tests: int = 5):
    """
    Testet das Modell durch Generierung von Text zu verschiedenen Prompts.
    """
    print("\n" + "="*60)
    print("✨ STEP 4: TEST-GENERIERUNG")
    print("="*60)
    
    test_prompts = [
        "künstliche intelligenz",
        "schwarze loch",
        "covid",
        "quantencomputer",
        "klimawandel",
    ][:num_tests]
    
    for prompt in test_prompts:
        print(f"\n  ❓ Prompt: '{prompt}'")
        
        # Generiere
        encoded_prompt = tokenizer.encode(prompt)
        idx = torch.tensor([encoded_prompt], dtype=torch.long).to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            for _ in range(100):
                idx_cond = idx[:, -BLOCK_SIZE:]
                logits = model(idx_cond)
                logits_last = logits[:, -1, :]
                
                top_k = 15
                top_logits, top_indices = torch.topk(logits_last, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                idx_next = top_indices.gather(-1, sampled_idx)
                
                idx = torch.cat([idx, idx_next], dim=1)
                
                if idx_next.item() == tokenizer.char2idx.get('\n', -1):
                    break
        
        generated_ids = idx[0].tolist()
        answer = tokenizer.decode(generated_ids)
        answer = answer[len(prompt):].strip()
        
        print(f"  💬 {answer[:80]}...")


# ===== TRAINING STATISTIKEN =====
def print_training_stats():
    """
    Zeigt die Trainings-Statistiken an.
    """
    if not os.path.exists(TRAINING_LOG):
        print("  (Noch kein Training durchgeführt)")
        return
    
    with open(TRAINING_LOG, "r") as f:
        log = json.load(f)
    
    print(f"\n  📊 Trainings-Statistik:")
    print(f"     • Gesamt Epochen: {log['total_epochs']}")
    print(f"     • Trainings-Runden: {len(log['history'])}")
    
    if log['history']:
        latest = log['history'][-1]
        print(f"     • Letzte Runde: {latest['round']}")
        print(f"     • Durchschnitt Loss: {latest['avg_loss']:.4f}")
        print(f"     • Zeitstempel: {latest['timestamp']}")


# ===== MAIN =====
def main():
    """
    Hauptfunktion: Automatisches Multi-Topic Training
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " MINI LLM AUTO-TRAINER: Multi-Topic Learning ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # STEP 1: Daten sammeln
        training_data = collect_multi_topic_data()
        
        if not training_data or len(training_data) < 500:
            print("\n❌ Nicht genug Trainingsdaten! Abbruch.")
            return
        
        # STEP 2: Modell setup
        model, tokenizer = setup_model(training_data, load_existing=True)
        
        # STEP 3: Training
        model, log = continuous_training(model, tokenizer, training_data, num_rounds=3)
        
        # STEP 4: Tests
        test_generation(model, tokenizer, num_tests=5)
        
        # SUMMARY
        print("\n" + "="*60)
        print("✅ AUTO-TRAINING ABGESCHLOSSEN!")
        print("="*60)
        
        print_training_stats()
        
        print(f"\n  📚 Trainierte Themen ({len(TRAINING_TOPICS)}):")
        for i, topic in enumerate(TRAINING_TOPICS, 1):
            print(f"     {i:2}. {topic}")
        
        print(f"\n  🤖 Modell: {MODEL_FILE}")
        print(f"  📝 Corpus: {len(training_data):,} Zeichen")
        print(f"  📊 Log: {TRAINING_LOG}")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
