# Mini LLM Training & Inference Engine

## Übersicht
Das Projekt **Mini LLM** ist ein Python-Framework zur Erstellung, zum Training und zur Evaluation eigener kleiner Transformer-Sprachmodelle. Das System bietet eine komplette Pipeline von der Tokenisierung bis zur interaktiven Generierung.

## Projektstruktur & Architektur
- `model/tiny_transformer.py`: Definition der PyTorch Transformer-Architektur.
- `utils/tokenizer.py`: Tokenizer-Implementierung zur Aufbereitung von Textdaten.
- `mega_train.py` & `auto_train.py`: Skripte zur Steuerung automatisierter Trainingsabläufe.
- `chat_with_llm.py`: Interaktives Befehlszeilen-Interface zur Textgenerierung.
- `compare_gpt.py`: Vergleichswerkzeug für Modellantworten.

## Hauptfunktionalitäten
- **Eigenes Transformer-Modell**: Implementierung einer leichten Sprachmodell-Architektur in PyTorch.
- **Automatisierte Trainingsschleifen**: Protokollierung von Verlustwerten in `training_log.json`.
- **Interaktiver Chat**: Generierung von Antworten über die Konsole.

## Ausführung & Nutzung
Das Training wird über `python mega_train.py` gestartet. Interaktive Testläufe erfolgen über `python chat_with_llm.py`.

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz.
