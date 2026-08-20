# utils/google_search.py

import requests

# HIER DEINE ECHTEN DATEN EINTRAGEN
GOOGLE_API_KEY = "AIzaSyDBjVUT_iGP9LBAiV5m2DKzFhr6CHNSY1U"
GOOGLE_CX = "82187612c138f4b21"


def google_search(query, num_results=3):
    """
    Führt eine Google Custom Search aus und gibt die besten Ergebnisse zurück.
    """
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num_results,
    }

    res = requests.get(url, params=params)
    res.raise_for_status()  # wirft Fehler, wenn HTTP nicht ok ist

    data = res.json()

    results = []

    if "items" in data:
        for item in data["items"]:
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link"),
            })

    return results


if __name__ == "__main__":
    # kleiner Test
    r = google_search("Warum ist der Himmel blau?", num_results=2)
    from pprint import pprint
    pprint(r)