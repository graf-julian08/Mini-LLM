# collect_google_data.py

from utils.google_search import google_search

def collect(topic, num_queries=5, top_k=5, outfile="data/google_corpus.txt"):
    """
    Holt mehrere Google-Suchen zu einem Thema und speichert die besten Snippets als Trainingsdaten.
    """
    all_texts = []

    queries = [
        topic,
        f"erklärung {topic}",
        f"was ist {topic}",
        f"info {topic}",
        f"{topic} einfach erklärt",
    ]

    for q in queries[:num_queries]:
        print(f"Suche: {q}")
        results = google_search(q, num_results=top_k)

        for r in results:
            title = r["title"]
            snippet = r["snippet"]

            # Kleines Cleaning
            clean = f"{title}. {snippet}".replace("\n", " ")

            all_texts.append(clean)

    # Anhängen an Datei
    with open(outfile, "a", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t + "\n")

    print(f"FERTIG! {len(all_texts)} Texts gespeichert in {outfile}")


if __name__ == "__main__":
    topic = input("Thema: ")
    collect(topic)