#!/usr/bin/env python3
"""
Mini LLM Pipeline: Prompt → Google-Daten → Training → Antwort
Alles in einem übersichtlichen Script!
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ===== IMPORTS =====
from utils.google_search import google_search
from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM


# ===== KONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FILE = "data/google_corpus.txt"
MODEL_FILE = "tiny_transformer.pt"

# Hyperparameter
BLOCK_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 512
DROPOUT = 0.1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15  # Mehr Epochen für besseres Lernen
BATCHES_PER_EPOCH = 50  # Mehr Batches pro Epoche
MAX_GENERATE_TOKENS = 150
MAX_TRAINING_ROUNDS = 3  # Automatische Trainings-Iterationen


# ===== STEP 1: GOOGLE-DATEN HOLEN =====
def fetch_google_data(prompt: str) -> str:
    """
    Holt Google-Daten zum Prompt und speichert sie als Trainingsdaten.
    Gibt den gesammelten Text zurück.
    """
    print("\n" + "="*60)
    print("📡 STEP 1: GOOGLE-DATEN HOLEN")
    print("="*60)
    
    # Verschiedene Suchanfragen zum Prompt
    queries = [
        prompt,
        f"erklärung {prompt}",
        f"was ist {prompt}",
        f"info {prompt}",
        f"{prompt} einfach erklärt",
    ]
    
    all_texts = []
    total_results = 0
    
    for q in queries:
        print(f"\n  🔍 Suche: '{q}'")
        try:
            results = google_search(q, num_results=5)
            
            for r in results:
                title = r["title"]
                snippet = r["snippet"]
                clean_text = f"{title}. {snippet}".replace("\n", " ")
                all_texts.append(clean_text)
                total_results += 1
            
            print(f"     ✓ {len(results)} Ergebnisse gefunden")
        except Exception as e:
            print(f"     ✗ Fehler: {e}")
    
    # Speichern
    collected_text = "\n".join(all_texts)
    
    # Neuen Corpus erstellen (nicht anhängen)
    os.makedirs("data", exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(collected_text)
    
    print(f"\n  ✅ {total_results} Texte gesammelt und gespeichert!")
    print(f"  📝 Datei: {CORPUS_FILE}")
    
    return collected_text


# ===== STEP 2: MODELL TRAINIEREN =====
def train_model(training_data: str) -> TinyTransformerLM:
    """
    Trainiert das Modell auf den Google-Daten.
    """
    print("\n" + "="*60)
    print("🤖 STEP 2: MODELL TRAINIEREN")
    print("="*60)
    
    # Tokenizer erstellen
    print(f"\n  📊 Tokenizer wird erstellt...")
    tokenizer = CharTokenizer(training_data)
    vocab_size = tokenizer.vocab_size
    encoded = torch.tensor(tokenizer.encode(training_data), dtype=torch.long)
    
    print(f"     • Vokabular-Größe: {vocab_size}")
    print(f"     • Text-Länge: {len(encoded):,} Tokens")
    
    # Modell erstellen
    print(f"\n  🏗️  Modell wird erstellt...")
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
    print(f"     • Parameter: {total_params:,}")
    print(f"     • Device: {DEVICE}")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training-Schleife
    print(f"\n  ⚙️  Training startet ({NUM_EPOCHS} Epochen)...")
    
    max_start = len(encoded) - BLOCK_SIZE - 1
    if max_start <= 0:
        print(f"     ⚠️  Text zu kurz! Verwende nur vorhandene Daten.")
        max_start = max(0, max_start)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for step in range(BATCHES_PER_EPOCH):  # Mehr Batches pro Epoche
            try:
                starts = torch.randint(0, max_start, (BATCH_SIZE,))
                x = torch.stack([encoded[i : i + BLOCK_SIZE] for i in starts])
                y = torch.stack([encoded[i + 1 : i + 1 + BLOCK_SIZE] for i in starts])
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(x)  # (batch_size, block_size, vocab_size)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"     ⚠️  Batch-Fehler: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"     • Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_loss:.4f}")
    
    model.eval()
    
    # Modell speichern
    print(f"\n  💾 Modell wird gespeichert...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
    }, MODEL_FILE)
    print(f"     ✓ Modell: {MODEL_FILE}")
    
    return model, tokenizer


# ===== STEP 3: ANTWORT GENERIEREN =====
def generate_answer(model: TinyTransformerLM, tokenizer: CharTokenizer, prompt: str) -> str:
    """
    Generiert eine Antwort basierend auf dem Prompt und Modell.
    """
    # Prompt enkodieren
    encoded_prompt = tokenizer.encode(prompt)
    idx = torch.tensor([encoded_prompt], dtype=torch.long).to(DEVICE)
    
    # Generieren
    model.eval()
    generated_length = 0
    with torch.no_grad():
        for step in range(MAX_GENERATE_TOKENS):
            # Kontext auf block_size kürzen
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Prediction
            logits = model(idx_cond)  # (1, T, vocab_size)
            logits_last = logits[:, -1, :]  # (1, vocab_size)
            
            # Top-k sampling für bessere Qualität
            top_k = 15
            top_logits, top_indices = torch.topk(logits_last, top_k)
            probs = torch.softmax(top_logits, dim=-1)
            
            # Sample aus Top-k
            sampled_idx = torch.multinomial(probs, num_samples=1)
            idx_next = top_indices.gather(-1, sampled_idx)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
            generated_length += 1
            
            # Stoppe bei Zeilenumbruch (natürlicher Endpunkt)
            if generated_length > 50 and idx_next.item() == tokenizer.char2idx.get('\n', -1):
                break
    
    # Dekodieren
    generated_ids = idx[0].tolist()
    answer = tokenizer.decode(generated_ids)
    
    # Nur die neu generierten Teile (nach dem Prompt)
    answer = answer[len(prompt):].strip()
    
    return answer


# ===== STEP 3B: QUALITÄT BEWERTEN =====
def evaluate_answer_quality(answer: str) -> tuple[float, str]:
    """
    Bewertet die Qualität der Antwort (0-100).
    Gibt Score und Feedback zurück.
    """
    score = 0
    feedback = []
    
    # Check 1: Länge (sollte nicht zu kurz sein)
    if len(answer) < 10:
        feedback.append("zu kurz")
    elif len(answer) < 30:
        score += 30
        feedback.append("sehr kurz")
    elif len(answer) < 80:
        score += 50
        feedback.append("kurz")
    else:
        score += 70
        feedback.append("gute Länge")
    
    # Check 2: Wiederholungen (schlecht)
    words = answer.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            feedback.append("zu viele Wiederholungen")
        elif unique_ratio < 0.7:
            score += 10
            feedback.append("etwas repetitiv")
        else:
            score += 20
            feedback.append("vielfältig")
    
    # Check 3: Kohärenz (Vorkommen von Wort-ähnlichen Mustern)
    coherent_chars = sum(1 for c in answer if c.isalpha() or c.isspace() or c in '.,!?')
    coherence_ratio = coherent_chars / max(len(answer), 1)
    if coherence_ratio > 0.8:
        score += 10
        feedback.append("kohärent")
    else:
        feedback.append("inkoherent")
    
    return min(score, 100), " | ".join(feedback)


# ===== STEP 3C: AUTOMATISCHES TRAINING =====
def iterative_training(training_data: str, initial_model: TinyTransformerLM, tokenizer: CharTokenizer):
    """
    Trainiert das Modell iterativ und testet es mit verschiedenen Fragen.
    Wenn die Qualität gut ist, stoppt es. Sonst trainiert es weiter.
    """
    print("\n" + "="*60)
    print("🔄 STEP 3: ITERATIVES TRAINING & EVALUATION")
    print("="*60)
    
    model = initial_model
    test_prompts = [
        "das größte schwarze loch",
        "größe schwarze loch",
        "schwarze löcher",
        "wie funktionieren schwarze löcher",
    ]
    
    for round_num in range(MAX_TRAINING_ROUNDS):
        print(f"\n  📊 Evaluierungs-Runde {round_num + 1}/{MAX_TRAINING_ROUNDS}")
        print(f"  {'-'*56}")
        
        all_scores = []
        
        # Test mit verschiedenen Prompts
        for test_prompt in test_prompts:
            answer = generate_answer(model, tokenizer, test_prompt)
            quality_score, feedback = evaluate_answer_quality(answer)
            all_scores.append(quality_score)
            
            print(f"\n    ❓ Prompt: '{test_prompt}'")
            print(f"    💬 Antwort: {answer[:60]}...")
            print(f"    ⭐ Score: {quality_score}/100 ({feedback})")
        
        avg_score = sum(all_scores) / len(all_scores)
        print(f"\n  📈 Durchschnitt: {avg_score:.1f}/100")
        
        # Wenn gut genug, stoppen
        if avg_score >= 60 or round_num == MAX_TRAINING_ROUNDS - 1:
            print(f"\n  ✅ Qualität ausreichend! Beende Training.")
            break
        
        # Sonst: Weiteres Training
        print(f"\n  🚀 Qualität nicht ausreichend. Trainiere weiter...")
        print(f"  ⚙️  Training (weitere {NUM_EPOCHS} Epochen)...")
        
        vocab_size = tokenizer.vocab_size
        encoded = torch.tensor(tokenizer.encode(training_data), dtype=torch.long)
        max_start = len(encoded) - BLOCK_SIZE - 1
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            num_batches = 0
            
            for step in range(BATCHES_PER_EPOCH):
                try:
                    starts = torch.randint(0, max_start, (BATCH_SIZE,))
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
            if (epoch + 1) % 5 == 0:
                print(f"     • Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_loss:.4f}")
        
        model.eval()
    
    return model


# ===== MAIN =====
def main():
    """
    Hauptfunktion: Alle Schritte nacheinander ausführen.
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " MINI LLM PIPELINE: Google → Training → Generierung ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    # Input vom User
    user_prompt = input("\n🎯 Gib einen Prompt ein: ").strip()
    
    if not user_prompt:
        print("❌ Leerer Prompt! Abbruch.")
        return
    
    try:
        # STEP 1
        training_data = fetch_google_data(user_prompt)
        
        if not training_data or len(training_data) < 100:
            print("\n❌ Nicht genug Trainingsdaten gesammelt! Abbruch.")
            return
        
        # STEP 2
        model, tokenizer = train_model(training_data)
        
        # STEP 3: Iteratives Training & Evaluation
        model = iterative_training(training_data, model, tokenizer)
        
        # Final Answer
        print("\n" + "="*60)
        print("✨ FINALE ANTWORT")
        print("="*60)
        
        print(f"\n  💭 Prompt: '{user_prompt}'")
        print(f"\n  ⏳ Generiere finale Antwort...")
        answer = generate_answer(model, tokenizer, user_prompt)
        
        print(f"\n  🤖 Antwort:\n")
        print(f"  {answer}")
        
        # SUMMARY
        print("\n" + "="*60)
        print("✅ FERTIG!")
        print("="*60)
        print(f"\n  📥 Input: {user_prompt}")
        print(f"  📊 Trainingsdaten: {len(training_data):,} Zeichen")
        print(f"  🤖 Modell: {MODEL_FILE}")
        print(f"  📤 Output: {answer[:100]}...")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
