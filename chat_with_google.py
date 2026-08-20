# chat_with_google.py

from utils.google_search import google_search
from generate_transformer import generate_text


def build_prompt(user_input, google_results):
    """
    Baut den Prompt für das LLM:
    - Kurzfassung der Google-Ergebnisse
    - Dann Chat-Struktur "User: ... / Bot:"
    """
    if not google_results:
        info = "Keine Ergebnisse von Google gefunden."
    else:
        lines = []
        for res in google_results[:3]:  # max 3 Ergebnisse
            line = f"- {res['title']}: {res['snippet']}"
            lines.append(line)
        info = "\n".join(lines)

    system_info = (
        "Hier sind Infos aus dem Internet:\n"
        f"{info}\n\n"
        "Benutze diese Infos, um eine kurze, verständliche Antwort zu geben.\n"
    )

    prompt = system_info + f"\nUser: {user_input}\nBot:"
    return prompt


def main():
    print("Mini-Transformer-LLM mit Google-Suche. Zum Beenden: Ctrl+C")
    while True:
        try:
            user_in = input("\nDu: ")
        except (EOFError, KeyboardInterrupt):
            print("\nCiao 👋")
            break

        if not user_in.strip():
            continue

        print("→ Suche bei Google...")
        google_data = google_search(user_in, num_results=3)

        prompt = build_prompt(user_in, google_data)
        print("→ Generiere Antwort…")
        answer = generate_text(prompt, max_new_tokens=300)

        # Optional: nur den Teil ab "Bot:" anzeigen
        if "Bot:" in answer:
            answer = answer.split("Bot:", 1)[1].strip()

        print("\nBot:", answer)


if __name__ == "__main__":
    main()