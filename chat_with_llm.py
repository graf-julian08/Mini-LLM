#!/usr/bin/env python3
"""
Mini LLM Chat Interface: Interaktive Konversation mit dem trainierten Modell
"""

import os
import torch
import torch.nn.functional as F
from utils.tokenizer import CharTokenizer
from model.tiny_transformer import TinyTransformerLM


# ===== KONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE = "tiny_transformer.pt"
CORPUS_FILE = "data/mega_corpus.txt"


class ChatBot:
    """
    Interaktives Chat-Interface für das LLM
    """
    
    def __init__(self):
        """Lade das trainierte Modell"""
        print("\n" + "="*60)
        print("🤖 MINI LLM CHAT BOT LADEN")
        print("="*60)
        
        if not os.path.exists(MODEL_FILE):
            print(f"\n❌ Fehler: {MODEL_FILE} nicht gefunden!")
            print(f"   Bitte erst trainieren: python mega_train.py")
            raise FileNotFoundError(MODEL_FILE)
        
        if not os.path.exists(CORPUS_FILE):
            print(f"\n❌ Fehler: {CORPUS_FILE} nicht gefunden!")
            raise FileNotFoundError(CORPUS_FILE)
        
        # Tokenizer laden
        print(f"\n  📖 Lade Tokenizer...")
        with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            corpus = f.read()
        
        self.tokenizer = CharTokenizer(corpus)
        print(f"     ✓ Vokabular: {self.tokenizer.vocab_size} Zeichen")
        
        # Modell laden
        print(f"  🤖 Lade Modell...")
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
        
        # Verwende die Größe aus dem Checkpoint
        saved_vocab_size = checkpoint.get("vocab_size", self.tokenizer.vocab_size)
        saved_block_size = checkpoint.get("block_size", 128)
        
        # Modell mit den korrekten Größen erstellen
        # Versuche zuerst mit den großen Parametern (für neue Modelle)
        try:
            self.model = TinyTransformerLM(
                vocab_size=saved_vocab_size,
                block_size=saved_block_size,
                embed_dim=512,
                num_heads=8,
                num_layers=6,
                ff_dim=2048,
                dropout=0.1,
            ).to(DEVICE)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            # Falls das nicht funktioniert, versuche mit kleinen Parametern
            print(f"     ⚠️  Große Modell-Parameter passen nicht, versuche kleinere...")
            self.model = TinyTransformerLM(
                vocab_size=saved_vocab_size,
                block_size=saved_block_size,
                embed_dim=256,
                num_heads=4,
                num_layers=2,
                ff_dim=512,
                dropout=0.1,
            ).to(DEVICE)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.model.eval()
        
        print(f"     ✓ Modell geladen")
        print(f"     ✓ Epochs trained: {checkpoint.get('epochs_trained', '?')}")
        print(f"     ✓ Device: {DEVICE}")
        
        self.block_size = saved_block_size
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.8) -> str:
        """
        Generiert Text basierend auf einem Prompt.
        
        Args:
            prompt: Eingabe-Text
            max_length: Maximale Länge der Ausgabe
            temperature: Kreativität (0.5 = konservativ, 1.0 = normal, 1.5 = kreativ)
        
        Returns:
            Generierter Text
        """
        # Prompt enkodieren
        encoded_prompt = self.tokenizer.encode(prompt)
        idx = torch.tensor([encoded_prompt], dtype=torch.long).to(DEVICE)
        
        # Generieren
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Kontext auf block_size kürzen
                idx_cond = idx[:, -self.block_size:]
                
                # Prediction
                logits = self.model(idx_cond)  # (1, T, vocab_size)
                logits_last = logits[:, -1, :]  # (1, vocab_size)
                
                # Temperature Scaling
                logits_last = logits_last / temperature
                
                # Top-k Sampling
                top_k = min(20, logits_last.shape[-1])
                top_logits, top_indices = torch.topk(logits_last, top_k, dim=-1)
                probs = F.softmax(top_logits, dim=-1)
                
                # Sample aus Top-k
                sampled_idx = torch.multinomial(probs, num_samples=1)  # (1, 1)
                idx_next = top_indices.gather(-1, sampled_idx)  # (1, 1)
                
                # Append
                idx = torch.cat([idx, idx_next], dim=1)
        
        # Dekodieren
        generated_ids = idx[0].tolist()
        answer = self.tokenizer.decode(generated_ids)
        
        # Nur den generierten Teil
        answer = answer[len(prompt):]
        
        return answer.strip()
    
    def chat(self):
        """Interaktive Chat-Schleife"""
        print("\n" + "="*60)
        print("💬 MINI LLM CHAT")
        print("="*60)
        print("\nStichwörter eingeben. Modell antwortet!")
        print("Befehle:")
        print("  /temp <wert>   - Kreativität ändern (0.5-2.0)")
        print("  /len <wert>    - Länge ändern (50-500)")
        print("  /help          - Hilfe anzeigen")
        print("  /quit oder /exit - Chat beenden")
        print("\n" + "-"*60 + "\n")
        
        max_length = 200
        temperature = 0.8
        
        while True:
            try:
                user_input = input("👤 Du: ").strip()
                
                if not user_input:
                    continue
                
                # Commands
                if user_input.lower() == "/quit" or user_input.lower() == "/exit":
                    print("\n👋 Auf Wiedersehen!\n")
                    break
                
                if user_input.lower() == "/help":
                    print("\n📖 Hilfe:")
                    print("  Gib einfach ein Stichwort ein, z.B.:")
                    print("  - 'künstliche intelligenz'")
                    print("  - 'python code'")
                    print("  - 'quantenmechanik'")
                    print("  - 'machine learning'\n")
                    continue
                
                if user_input.startswith("/temp "):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"✓ Kreativität auf {temperature} gesetzt\n")
                    except:
                        print("❌ Ungültige Eingabe\n")
                    continue
                
                if user_input.startswith("/len "):
                    try:
                        max_length = int(user_input.split()[1])
                        max_length = max(50, min(500, max_length))
                        print(f"✓ Länge auf {max_length} gesetzt\n")
                    except:
                        print("❌ Ungültige Eingabe\n")
                    continue
                
                # Generiere Antwort
                print("\n🤖 Bot: ", end="", flush=True)
                
                try:
                    response = self.generate(user_input, max_length, temperature)
                    print(response)
                    print()
                
                except Exception as e:
                    print(f"\n❌ Fehler: {e}\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Chat beendet.\n")
                break
            except Exception as e:
                print(f"\n❌ Fehler: {e}\n")


class Analyzer:
    """
    Analysiert das trainierte Modell und zeigt Statistiken
    """
    
    @staticmethod
    def show_stats():
        """Zeigt Modell-Statistiken an"""
        print("\n" + "="*60)
        print("📊 MODELL-STATISTIKEN")
        print("="*60)
        
        if not os.path.exists(MODEL_FILE):
            print(f"\n❌ Modell nicht gefunden: {MODEL_FILE}")
            return
        
        checkpoint = torch.load(MODEL_FILE, map_location="cpu")
        
        print(f"\n  📈 Training Info:")
        print(f"     • Epochs trained: {checkpoint.get('epochs_trained', '?')}")
        print(f"     • Block size: {checkpoint.get('block_size', '?')}")
        print(f"     • Vocab size: {checkpoint.get('vocab_size', '?')}")
        
        # Parameter zählen
        total_params = 0
        state_dict = checkpoint["model_state_dict"]
        for key, tensor in state_dict.items():
            params = tensor.numel()
            total_params += params
            if params > 100000:
                print(f"     • {key}: {params:,}")
        
        print(f"\n  🧠 Modell-Größe:")
        print(f"     • Parameter gesamt: {total_params:,}")
        print(f"     • Speicher (MB): {total_params * 4 / (1024*1024):.1f}")
        
        # Corpus Info
        if os.path.exists(CORPUS_FILE):
            corpus_size = os.path.getsize(CORPUS_FILE)
            print(f"\n  📚 Corpus:")
            print(f"     • Datei-Größe: {corpus_size / (1024*1024):.1f}MB")
        
        # Training Log
        import json
        if os.path.exists("training_log.json"):
            with open("training_log.json", "r") as f:
                log = json.load(f)
            
            print(f"\n  📊 Training Log:")
            print(f"     • Trainings-Runden: {len(log.get('history', []))}")
            if log.get('history'):
                latest = log['history'][-1]
                print(f"     • Letzte Loss: {latest.get('avg_loss', '?'):.4f}")
                corpus_size = latest.get('corpus_size_mb', '?')
                if isinstance(corpus_size, (int, float)):
                    print(f"     • Corpus-Größe: {corpus_size:.1f}MB")
                else:
                    print(f"     • Corpus-Größe: {corpus_size}")


def main():
    """Hauptfunktion"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " MINI LLM - Chat & Interaktion ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    print("\nWas möchtest du tun?")
    print("  1. 💬 Chat mit dem Modell")
    print("  2. 📊 Modell-Statistiken anzeigen")
    print("  3. 🚀 Beende Programm")
    
    choice = input("\nWahl (1-3): ").strip()
    
    if choice == "1":
        try:
            bot = ChatBot()
            bot.chat()
        except Exception as e:
            print(f"\n❌ Fehler beim Laden: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "2":
        Analyzer.show_stats()
    
    elif choice == "3":
        print("\nAuf Wiedersehen! 👋\n")
    
    else:
        print("\n❌ Ungültige Eingabe\n")


if __name__ == "__main__":
    # Fix für die Zeile oben
    model = None  # wird in ChatBot geladen
    main()
