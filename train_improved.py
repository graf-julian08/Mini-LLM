#!/usr/bin/env python3
"""
Mini LLM Improved Training: Better data + training for coherent output
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime

from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM


# ===== KONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FILE = "data/mega_corpus.txt"
MODEL_FILE = "tiny_transformer.pt"
TRAINING_LOG = "training_log.json"

# Hyperparameter - Optimiert für bessere Qualität
BLOCK_SIZE = 256  # Größer = mehr Kontext
BATCH_SIZE = 16  # Kleiner = stabiler
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 4
FF_DIM = 1536
DROPOUT = 0.05  # Weniger Dropout für kleinere Modelle
LEARNING_RATE = 5e-5  # Noch kleinere LR
NUM_EPOCHS = 20  # Mehr Epochen
BATCHES_PER_EPOCH = 100

CURRENT_CORPUS_SIZE_LIMIT = 1024 * 1024 * 1024  # 1GB


# ===== BESSERER CORPUS BUILDER =====
def build_quality_corpus() -> str:
    """
    Baut einen qualitativ hochwertigen Corpus mit strukturierten Daten.
    """
    print("\n" + "="*60)
    print("📥 STEP 1: QUALITÄTS-CORPUS AUFBAUEN")
    print("="*60)
    
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(CORPUS_FILE):
        size = os.path.getsize(CORPUS_FILE)
        if size > 50 * 1024 * 1024:  # > 50MB
            print(f"\n  ✓ Corpus existiert: {size / (1024*1024):.1f}MB")
            with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    
    print(f"\n  🏗️  Erstelle neuen Corpus...")
    
    all_texts = []
    
    # High-Quality Content mit guter Struktur
    knowledge_base = """
Solar energy is renewable energy from the sun. Photovoltaic cells convert sunlight into electricity. 
Solar panels are widely used for residential and commercial applications. The sun provides abundant 
clean energy. Solar technology continues to improve and become more efficient.

Artificial Intelligence (AI) is transforming industries and society. Machine learning enables systems 
to learn from data. Deep learning uses neural networks with multiple layers. AI applications include 
image recognition, natural language processing, and autonomous systems.

Quantum computers use quantum bits (qubits) for computation. They exploit quantum superposition and 
entanglement. Quantum computing promises exponential speedup for certain problems. Current quantum 
computers are in early stages of development.

Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities 
increase greenhouse gas emissions. Carbon dioxide and methane trap heat in the atmosphere. Mitigation 
requires renewable energy adoption and sustainable practices.

Biotechnology uses biological systems for practical applications. CRISPR gene editing offers precision 
genetic modifications. Synthetic biology designs new biological systems. Biopharmaceuticals include 
vaccines and therapeutic proteins.

Neuroscience studies the brain and nervous system. Neural networks process information through synaptic 
connections. Neurotransmitters enable communication between neurons. Brain imaging reveals structure 
and function. Cognitive science integrates neuroscience with psychology.

Blockchain technology enables distributed ledgers and decentralized systems. Cryptographic hashing 
ensures data integrity. Smart contracts automate agreement execution. Cryptocurrency applications 
include Bitcoin and Ethereum.

Space exploration expands human knowledge and capabilities. Spacecraft carry astronauts and instruments 
to orbit and beyond. Satellite technology enables global communication and observation. Mars missions 
seek signs of past habitability.

Renewable energy sources include solar, wind, hydroelectric, and geothermal. Energy efficiency reduces 
consumption. Grid modernization enables distributed generation. Storage technologies like batteries 
support intermittent renewables.

Chemistry studies matter, reactions, and molecular interactions. The periodic table organizes elements. 
Chemical bonds include ionic, covalent, and metallic. Reactions follow conservation of mass and energy 
principles.

Physics explores fundamental forces and particles. Classical mechanics describes macroscopic motion. 
Quantum mechanics governs atomic behavior. Relativity connects space, time, mass, and energy. 
Thermodynamics addresses heat and entropy.

Biology studies living organisms and life processes. Cells are basic units of life. Genetics explains 
heredity through DNA. Evolution shapes species through natural selection. Ecology examines ecosystems.

Computer Science provides theoretical foundations for computation. Algorithms solve problems efficiently. 
Data structures organize information. Programming languages express computational logic. Artificial 
intelligence extends computational capabilities.

Medicine diagnoses and treats diseases. Anatomy maps body structures. Physiology explains function. 
Pharmacology studies drugs and effects. Surgery removes or repairs diseased tissues. Preventive medicine 
maintains health.

Engineering applies scientific principles to practical problems. Civil engineering builds infrastructure. 
Mechanical engineering designs machines. Electrical engineering manages power and electronics. Software 
engineering develops applications.

Economics studies production, distribution, and consumption. Markets allocate resources through prices. 
Supply and demand determine equilibrium. Macroeconomics examines aggregate economies. Microeconomics 
studies individual decisions.

Psychology explores mind and behavior. Cognition involves perception, memory, and thinking. Emotion 
affects motivation and well-being. Social psychology examines group behavior. Clinical psychology 
treats mental disorders.

Philosophy asks fundamental questions about reality and knowledge. Epistemology questions what we can 
know. Metaphysics addresses what exists. Ethics examines right and wrong. Logic provides reasoning rules.

History documents past events and their significance. Ancient civilizations developed writing and 
governance. Medieval periods saw feudalism and religious authority. Modern era brought scientific 
revolution and industrialization. Contemporary history addresses recent events.

Literature expresses human experience through narrative and poetry. Fiction creates imaginary worlds. 
Poetry uses concise language and imagery. Drama presents stories through character dialogue. 
Non-fiction documents facts and ideas.

Art expresses creativity through visual and other mediums. Painting uses color and form. Sculpture 
shapes three-dimensional space. Music combines sounds and rhythm. Dance expresses movement.

Music combines sounds in organized patterns. Melody creates memorable sequences. Harmony combines 
simultaneous notes. Rhythm provides temporal structure. Instrumentation produces diverse sounds.

Architecture designs buildings and spaces. Structural systems support weight. Aesthetics create visual 
appeal. Functionality meets practical needs. Materials include wood, stone, steel, and concrete.

Mathematics provides language for quantitative reasoning. Arithmetic performs basic calculations. 
Algebra uses symbols to represent quantities. Geometry studies shapes and space. Calculus analyzes 
change and accumulation.

Statistics analyzes data to draw conclusions. Probability measures likelihood of events. Distributions 
describe data patterns. Hypothesis testing evaluates claims. Regression models relationships.

Information Technology manages computer systems and networks. Databases store organized data. Networks 
connect computers. Cybersecurity protects against threats. Cloud computing provides remote resources.

Environmental Science studies Earth systems and conservation. Ecology examines species relationships. 
Geology studies rocks and earth processes. Meteorology forecasts weather. Conservation protects natural 
resources.

Agriculture produces food through cultivation. Crops require soil, water, and sunlight. Livestock 
provides meat, dairy, and fiber. Sustainable practices maintain productivity. Technology improves yields.

Transportation moves people and goods. Vehicles include cars, trains, ships, and aircraft. Infrastructure 
includes roads, rails, and ports. Logistics optimizes movement. Alternative fuels reduce emissions.
"""
    
    # Wiederhole Content mit Variationen für besseres Lernen
    for i in range(10):
        all_texts.append(knowledge_base)
        if (i + 1) % 3 == 0:
            print(f"     • Content repetition {i+1}/10")
    
    # Kombiniere alles
    mega_corpus = "\n\n".join(all_texts)
    
    # Speichern
    print(f"\n  💾 Speichere Corpus...")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(mega_corpus)
    
    corpus_size_mb = len(mega_corpus.encode('utf-8')) / (1024*1024)
    print(f"     ✓ Corpus: {corpus_size_mb:.1f}MB")
    
    return mega_corpus


# ===== MODELL SETUP =====
def setup_model(training_data: str, load_existing: bool = True) -> tuple:
    """Erstellt oder lädt das Modell mit besseren Parametern."""
    print("\n" + "="*60)
    print("🤖 STEP 2: MODELL SETUP")
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
                saved_config = checkpoint.get("config", {})
                
                model = TinyTransformerLM(
                    vocab_size=vocab_size,
                    block_size=saved_config.get("block_size", BLOCK_SIZE),
                    embed_dim=saved_config.get("embed_dim", EMBED_DIM),
                    num_heads=saved_config.get("num_heads", NUM_HEADS),
                    num_layers=saved_config.get("num_layers", NUM_LAYERS),
                    ff_dim=saved_config.get("ff_dim", FF_DIM),
                    dropout=saved_config.get("dropout", DROPOUT),
                ).to(DEVICE)
                
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"     ✓ Modell geladen ({checkpoint.get('epochs_trained', 0)} Epochen)")
                model_loaded = True
        except Exception as e:
            print(f"     ⚠️  Fehler: {e}")
    
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
    
    return model, tokenizer


# ===== TRAINING =====
def train_model(model: TinyTransformerLM, tokenizer: CharTokenizer, training_data: str):
    """Trainiert das Modell mit verbesserter Stabilität."""
    print("\n" + "="*60)
    print("⚙️  STEP 3: INTENSIVES TRAINING")
    print("="*60)
    
    # Log laden
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, "r") as f:
            log = json.load(f)
    else:
        log = {"total_epochs": 0, "history": []}
    
    vocab_size = tokenizer.vocab_size
    encoded = torch.tensor(tokenizer.encode(training_data), dtype=torch.long)
    max_start = len(encoded) - BLOCK_SIZE - 1
    
    print(f"\n  📊 Training Setup:")
    print(f"     • Encoded Length: {len(encoded):,}")
    print(f"     • Block Size: {BLOCK_SIZE}")
    print(f"     • Batch Size: {BATCH_SIZE}")
    print(f"     • Epochs: {NUM_EPOCHS}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * BATCHES_PER_EPOCH)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
    print(f"\n  🚀 Training startet...")
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (step + 1) % 25 == 0:
                    print(f"     Epoch {epoch+1} | Batch {step+1}/{BATCHES_PER_EPOCH}: Loss = {total_loss/num_batches:.4f}")
                    
            except Exception as e:
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        all_losses.append(avg_loss)
        print(f"  ✓ Epoch {epoch+1}/{NUM_EPOCHS}: Avg Loss = {avg_loss:.4f}")
    
    # Log speichern
    log["total_epochs"] += NUM_EPOCHS
    log["history"].append({
        "timestamp": datetime.now().isoformat(),
        "avg_loss": sum(all_losses) / len(all_losses),
        "final_loss": all_losses[-1] if all_losses else None,
    })
    
    # Modell speichern mit Konfiguration
    print(f"\n  💾 Speichere Modell...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "epochs_trained": log["total_epochs"],
        "config": {
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
        }
    }, MODEL_FILE)
    
    with open(TRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"     ✓ Modell gespeichert")
    
    model.eval()
    return model, log


def main():
    """Hauptfunktion"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " MINI LLM IMPROVED TRAINER ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # STEP 1: Corpus
        training_data = build_quality_corpus()
        
        # STEP 2: Modell
        model, tokenizer = setup_model(training_data, load_existing=True)
        
        # STEP 3: Training
        model, log = train_model(model, tokenizer, training_data)
        
        # SUMMARY
        print("\n" + "="*60)
        print("✅ TRAINING ABGESCHLOSSEN!")
        print("="*60)
        
        print(f"\n  📊 Statistik:")
        print(f"     • Gesamt Epochen: {log['total_epochs']}")
        print(f"     • Final Loss: {log['history'][-1]['final_loss']:.4f}")
        print(f"     • Modell: {MODEL_FILE}")
        
        print(f"\n  💡 Nächster Schritt:")
        print(f"     python chat_with_llm.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
