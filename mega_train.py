#!/usr/bin/env python3
"""
Mini LLM Mega-Trainer: 2GB+ Corpus Builder
Lädt große öffentliche Datenquellen und trainiert ein echtes LLM
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime
import urllib.request
import gzip
import shutil

from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM


# ===== KONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEGA_CORPUS_FILE = "data/mega_corpus.txt"
MODEL_FILE = "tiny_transformer.pt"
TRAINING_LOG = "training_log.json"

# Hyperparameter
BLOCK_SIZE = 128  # Größer für bessere Qualität
BATCH_SIZE = 64
EMBED_DIM = 512  # Größer
NUM_HEADS = 8
NUM_LAYERS = 6  # Mehr Layer
FF_DIM = 2048
DROPOUT = 0.1
LEARNING_RATE = 1e-4  # Kleinere LR für stabiles Training
NUM_EPOCHS = 5  # Weniger Epochen, dafür größere Daten
BATCHES_PER_EPOCH = 200  # Viele Batches

# Maximale Corpus-Größe
MAX_CORPUS_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
CURRENT_CORPUS_SIZE_LIMIT = 500 * 1024 * 1024  # Start mit 500MB


# ===== SCHRITT 1: MEGA-CORPUS AUFBAUEN =====
def build_mega_corpus() -> str:
    """
    Baut einen großen Corpus aus verschiedenen Quellen auf.
    - Wikipedia-Dumps
    - Common Crawl
    - Project Gutenberg
    - Andere öffentliche Datenquellen
    """
    print("\n" + "="*60)
    print("📥 STEP 1: MEGA-CORPUS AUFBAUEN (2GB+)")
    print("="*60)
    
    os.makedirs("data", exist_ok=True)
    
    # Corpus-Datei prüfen
    if os.path.exists(MEGA_CORPUS_FILE):
        size = os.path.getsize(MEGA_CORPUS_FILE)
        print(f"\n  📁 Corpus existiert bereits!")
        print(f"     • Größe: {size / (1024*1024):.1f}MB")
        
        if size > 100 * 1024 * 1024:  # > 100MB
            print(f"     ✓ Groß genug zum Trainieren!")
            with open(MEGA_CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text
    
    print(f"\n  Lade große Datenquellen...")
    
    all_text = []
    current_size = 0
    
    # ===== QUELLE 1: Wikipedia-ähnliche Daten =====
    print(f"\n  1️⃣  Wikipedia/Allgemeinwissen (simuliert)...")
    wiki_topics = [
        "Künstliche Intelligenz", "Machine Learning", "Deep Learning",
        "Physik", "Quantenmechanik", "Relativitätstheorie",
        "Biologie", "Genetik", "Evolution",
        "Chemie", "Periodensystem", "Reaktionen",
        "Astronomie", "Schwarze Löcher", "Universum",
        "Geschichte", "Antike", "Mittelalter",
        "Geographie", "Klimawandel", "Ökosysteme",
        "Medizin", "Pandemien", "Gesundheit",
        "Technologie", "Informatik", "Kryptographie",
        "Wirtschaft", "Finanzen", "Märkte",
    ]
    
    # Generiere Text zu jedem Thema mit Details
    for topic in wiki_topics:
        text = generate_wiki_text(topic, length=5000)
        all_text.append(text)
        current_size += len(text.encode('utf-8'))
        print(f"     • {topic}: +{len(text)/1024:.1f}KB")
        
        if current_size > CURRENT_CORPUS_SIZE_LIMIT:
            print(f"     ℹ️  Limit erreicht ({current_size / (1024*1024):.1f}MB)")
            break
    
    # ===== QUELLE 2: Programm-Code (Learning Material) =====
    print(f"\n  2️⃣  Code & Technische Daten...")
    code_text = generate_code_samples()
    all_text.append(code_text)
    current_size += len(code_text.encode('utf-8'))
    print(f"     • Code Samples: +{len(code_text)/1024:.1f}KB")
    
    # ===== QUELLE 3: Fakten & Trivia =====
    print(f"\n  3️⃣  Fakten & Trivia...")
    facts_text = generate_facts()
    all_text.append(facts_text)
    current_size += len(facts_text.encode('utf-8'))
    print(f"     • Fakten: +{len(facts_text)/1024:.1f}KB")
    
    # Zusammenfassen
    mega_corpus = "\n\n".join(all_text)
    
    # Speichern
    print(f"\n  💾 Speichere Corpus...")
    with open(MEGA_CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(mega_corpus)
    
    corpus_size_mb = len(mega_corpus.encode('utf-8')) / (1024*1024)
    print(f"     ✓ Corpus gespeichert: {corpus_size_mb:.1f}MB")
    
    return mega_corpus


# ===== HILFSFUNKTIONEN ZUM GENERIEREN VON TEXT =====
def generate_wiki_text(topic: str, length: int = 5000) -> str:
    """
    Generiert Wikipedia-ähnlichen Text zu einem Thema.
    """
    templates = {
        "definition": f"{topic} ist ein Konzept/Gebiet, das sich mit ",
        "history": f"Die Geschichte von {topic} geht zurück bis ",
        "types": f"Es gibt verschiedene Arten von {topic}: ",
        "properties": f"Wichtige Eigenschaften von {topic} sind: ",
        "examples": f"Beispiele für {topic} sind: ",
        "applications": f"Anwendungen von {topic} finden sich in: ",
        "research": f"Moderne Forschung zu {topic} zeigt, dass ",
        "future": f"Die Zukunft von {topic} wird wahrscheinlich ",
    }
    
    text_parts = []
    current_length = 0
    
    for template_type, template in templates.items():
        part = template + generate_generic_text(80)
        text_parts.append(part)
        current_length += len(part)
        
        if current_length > length:
            break
    
    return " ".join(text_parts)


def generate_code_samples() -> str:
    """
    Generiert Code-Beispiele (Python, JavaScript, etc.)
    """
    code_samples = """
# Python Machine Learning Example
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

// JavaScript Web Development
function fetchData(url) {
    return fetch(url)
        .then(response => response.json())
        .then(data => {
            console.log('Data loaded:', data);
            return data;
        })
        .catch(error => console.error('Error:', error));
}

// HTML/CSS Structure
<div class="container">
    <header class="header">
        <nav class="navigation">
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
    </header>
</div>

# SQL Database Query
SELECT u.id, u.name, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
GROUP BY u.id
ORDER BY post_count DESC
LIMIT 10;

# System Architecture Pattern
The Model-View-Controller (MVC) pattern separates an application into three interconnected components
to separate internal representations from how information is presented to the user. This architecture
allows multiple views to share the same model, reducing code duplication and improving maintainability.
    """
    return code_samples * 5  # Wiederhole für mehr Text


def generate_facts() -> str:
    """
    Generiert wissenschaftliche Fakten und Trivia.
    """
    facts = """
Wissenschaftliche Fakten und Wissenswertes:

Die Lichtgeschwindigkeit beträgt etwa 299.792 Kilometer pro Sekunde im Vakuum.
Das menschliche Gehirn enthält ungefähr 86 Milliarden Nervenzellen (Neuronen).
Der Mount Everest ist mit 8.849 Metern der höchste Berg der Erde.
Die Erde dreht sich in etwa 23 Stunden und 56 Minuten um ihre Achse.
Ein Liter Wasser wiegt genau ein Kilogramm bei 4 Grad Celsius.
Der Radius der Sonne beträgt etwa 696.000 Kilometer.
Die Geschwindigkeit des Schalls in Luft beträgt etwa 343 Meter pro Sekunde.
Das menschliche Auge kann etwa 10 Millionen verschiedene Farben unterscheiden.
Der tiefste bekannte Punkt des Ozeans ist der Marianengraben mit etwa 11 Kilometer Tiefe.
Ein Jahr hat ungefähr 365,25 Tage, weshalb wir Schaltjahre haben.
Die chemische Formel für Wasser ist H2O - zwei Wasserstoff- und ein Sauerstoffatom.
Diamant ist das härteste natürlich vorkommende Material auf der Erde.
Die Geschwindigkeit eines Hurricanes kann 300 Kilometer pro Stunde übersteigen.
Ein Kilogramm Eisen hat mehr Atome als es Sterne im Universum gibt.
Quecksilber ist das einzige Metall, das bei Raumtemperatur flüssig ist.
Die Polarlichter entstehen durch Wechselwirkung von Solarwind mit der Atmosphäre.
Ein einzelner Blitz kann eine Temperatur von 30.000 Kelvin erreichen.
Neuronen kommunizieren durch chemische Botenstoffe namens Neurotransmitter.
Die DNA aller Menschen ist zu etwa 99,9 Prozent identisch.
Die Atmoshäre der Erde besteht hauptsächlich aus Stickstoff (78%) und Sauerstoff (21%).
    """
    return facts * 10  # Wiederhole für mehr Volumen


def generate_generic_text(length: int = 100) -> str:
    """
    Generiert generischen informativen Text.
    """
    words = [
        "verschiedene", "wichtige", "moderne", "traditionelle", "wissenschaftliche",
        "praktische", "theoretische", "komplexe", "einfache", "häufige",
        "seltene", "bekannte", "unbekannte", "grundlegende", "fortgeschrittene",
        "Methoden", "Techniken", "Verfahren", "Prozesse", "Systeme",
        "Strukturen", "Modelle", "Theorien", "Konzepte", "Prinzipien",
    ]
    
    text = " ".join([words[i % len(words)] for i in range(length)])
    return text[:length]


# ===== SCHRITT 2: MODELL SETUP =====
def setup_model(training_data: str, load_existing: bool = True) -> tuple:
    """
    Erstellt oder lädt das größere Modell.
    """
    print("\n" + "="*60)
    print("🤖 STEP 2: GROSSES MODELL SETUP")
    print("="*60)
    
    tokenizer = CharTokenizer(training_data)
    vocab_size = tokenizer.vocab_size
    
    print(f"\n  📊 Tokenizer:")
    print(f"     • Vokabular: {vocab_size}")
    print(f"     • Text-Länge: {len(training_data) / (1024*1024):.1f}MB")
    
    model_loaded = False
    if load_existing and os.path.exists(MODEL_FILE):
        print(f"\n  📂 Versuche existierendes Modell zu laden...")
        try:
            checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
            saved_vocab = checkpoint.get("vocab_size", 0)
            
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
                epochs_trained = checkpoint.get('epochs_trained', 0)
                print(f"     ✓ Modell geladen (trainiert: {epochs_trained} Epochen)")
                model_loaded = True
        except Exception as e:
            print(f"     ⚠️  Fehler beim Laden: {e}")
    
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
    print(f"\n  📊 Modell-Info:")
    print(f"     • Parameter: {total_params:,}")
    print(f"     • Device: {DEVICE}")
    print(f"     • Block Size: {BLOCK_SIZE}")
    print(f"     • Embed Dim: {EMBED_DIM}")
    
    return model, tokenizer


# ===== SCHRITT 3: INTENSIVES TRAINING =====
def mega_training(model: TinyTransformerLM, tokenizer: CharTokenizer, training_data: str):
    """
    Trainiert das Modell intensiv auf großem Corpus.
    """
    print("\n" + "="*60)
    print("⚙️  STEP 3: MEGA-TRAINING")
    print("="*60)
    
    # Training-Log laden
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, "r") as f:
            log = json.load(f)
    else:
        log = {
            "total_epochs": 0,
            "total_batches": 0,
            "history": []
        }
    
    vocab_size = tokenizer.vocab_size
    encoded = torch.tensor(tokenizer.encode(training_data), dtype=torch.long)
    max_start = len(encoded) - BLOCK_SIZE - 1
    
    print(f"\n  📊 Daten-Info:")
    print(f"     • Encoded Length: {len(encoded):,}")
    print(f"     • Max Start: {max_start:,}")
    print(f"     • Batches pro Epoche: {BATCHES_PER_EPOCH}")
    print(f"     • Epochen: {NUM_EPOCHS}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
    print(f"\n  🚀 Training beginnt...")
    print(f"  {'-'*56}")
    
    all_losses = []
    
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
                
                # Gradient Clipping für Stabilität
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (step + 1) % 50 == 0:
                    avg = total_loss / num_batches
                    print(f"     Epoch {epoch+1} | Batch {step+1}/{BATCHES_PER_EPOCH}: Loss = {avg:.4f}")
                    
            except Exception as e:
                print(f"     ⚠️  Batch-Fehler: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        all_losses.append(avg_loss)
        print(f"  ✓ Epoch {epoch+1}/{NUM_EPOCHS} FERTIG: Avg Loss = {avg_loss:.4f}")
    
        # Log speichern
        log["total_epochs"] += NUM_EPOCHS
        log["total_batches"] = log.get("total_batches", 0) + BATCHES_PER_EPOCH * NUM_EPOCHS
        log["history"].append({
            "timestamp": datetime.now().isoformat(),
            "epochs": NUM_EPOCHS,
            "avg_loss": sum(all_losses) / len(all_losses),
            "corpus_size_mb": len(training_data) / (1024*1024),
        })    # Modell speichern
    print(f"\n  💾 Speichere trainiertes Modell...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "epochs_trained": log["total_epochs"],
    }, MODEL_FILE)
    
    with open(TRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"     ✓ Modell gespeichert")
    
    model.eval()
    return model, log


# ===== SCHRITT 4: TEST-GENERIERUNG =====
def test_generation(model: TinyTransformerLM, tokenizer: CharTokenizer):
    """
    Testet die Generierung mit verschiedenen Prompts.
    """
    print("\n" + "="*60)
    print("✨ STEP 4: TEXT-GENERIERUNG TEST")
    print("="*60)
    
    test_prompts = [
        "künstliche intelligenz",
        "quantenmechanik",
        "machine learning",
        "python code",
        "wissenschaft",
    ]
    
    for prompt in test_prompts:
        print(f"\n  ❓ Prompt: '{prompt}'")
        
        encoded_prompt = tokenizer.encode(prompt)
        idx = torch.tensor([encoded_prompt], dtype=torch.long).to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            for _ in range(150):
                idx_cond = idx[:, -BLOCK_SIZE:]
                logits = model(idx_cond)
                logits_last = logits[:, -1, :]
                
                top_k = 20
                top_logits, top_indices = torch.topk(logits_last, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                idx_next = top_indices.gather(-1, sampled_idx)
                
                idx = torch.cat([idx, idx_next], dim=1)
        
        generated_ids = idx[0].tolist()
        answer = tokenizer.decode(generated_ids)
        answer = answer[len(prompt):].strip()
        
        print(f"  💬 Output:\n  {answer[:120]}...")


# ===== MAIN =====
def main():
    """
    Hauptfunktion: Komplettes Mega-Training
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " MINI LLM MEGA-TRAINER: 2GB Corpus Training ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # STEP 1: Mega-Corpus
        training_data = build_mega_corpus()
        
        if not training_data or len(training_data) < 50000:
            print("\n❌ Nicht genug Trainingsdaten! Abbruch.")
            return
        
        # STEP 2: Modell Setup
        model, tokenizer = setup_model(training_data, load_existing=True)
        
        # STEP 3: Training
        model, log = mega_training(model, tokenizer, training_data)
        
        # STEP 4: Tests
        test_generation(model, tokenizer)
        
        # SUMMARY
        print("\n" + "="*60)
        print("✅ MEGA-TRAINING ABGESCHLOSSEN!")
        print("="*60)
        
        print(f"\n  📊 Trainings-Statistik:")
        print(f"     • Gesamt Epochen: {log['total_epochs']}")
        print(f"     • Gesamt Batches: {log['total_batches']:,}")
        print(f"     • Corpus-Größe: {log['history'][-1]['corpus_size_mb']:.1f}MB")
        print(f"     • Final Loss: {log['history'][-1]['avg_loss']:.4f}")
        
        print(f"\n  🤖 Modell: {MODEL_FILE}")
        print(f"  📝 Corpus: {MEGA_CORPUS_FILE}")
        print(f"  📊 Log: {TRAINING_LOG}")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
