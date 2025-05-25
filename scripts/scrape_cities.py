import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
import re
import os
import json

with open("data/state_articles.json", "r") as f:
    ARTICLE_URLS = json.load(f)


def normalize(text):
    return unidecode(text.strip().lower())

def extract_cities_from_article(state, url):
    city_data = []
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        for h2 in soup.find_all('h2'):
            heading = h2.get_text(strip=True)

            if not re.match(r'^\d+\.', heading):
                continue

            city_name = re.sub(r'^\d+\.\s*', '', heading).strip()

            p = h2.find_next_sibling('p')
            if p:
                description = p.get_text(strip=True)
                city_data.append({
                    "Location": city_name,
                    "state": state,
                    "description": description,
                    "source": "crazytourist",
                    "article_url": url
                })
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
    return city_data

def main():
    all_cities = []

    for state, urls in tqdm(ARTICLE_URLS.items(), desc="Scraping US states"):
        for url in urls:
            print(f"📍 Scraping {url} for {state}...")
            cities = extract_cities_from_article(state, url)
            all_cities.extend(cities)

    df = pd.DataFrame(all_cities)
    df.drop_duplicates(subset=["Location", "state"], inplace=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/us_locations_vibes.csv", index=False)
    print(f"\n✅ Saved {len(df)} U.S. vibe-rich cities to data/us_city_vibes.csv")

if __name__ == "__main__":
    main()