import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

SEARCH_KEYWORD = "Dollar Rp18.000"
MAX_ARTICLES = 3

def scrape_detik(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Gagal mengambil halaman:", e)
        return None

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    title_tag = soup.find("h1")
    title = (
        title_tag.get_text(strip=True)
        if title_tag
        else "No Title"
    )

    paragraphs = []
    article_container = soup.find(
        "div",
        class_="detail__body-text"
    )

    if article_container:
        for p in article_container.find_all("p"):
            text = p.get_text(
                separator=" ",
                strip=True
            )
            if len(text) > 30:
                paragraphs.append(text)

    article = " ".join(paragraphs)

    df = pd.DataFrame({
        "source": ["Detik"],
        "title": [title],
        "content": [article]
    })

    return df

def scrape_multiple_articles(keyword=SEARCH_KEYWORD, max_articles=MAX_ARTICLES):
    print(f"\n🔍 Mencari artikel tentang '{keyword}'...")

    # Cari dari search result
    search_url = f"https://finance.detik.com/search?q={keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Gagal mencari artikel: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Cari link artikel
    article_links = []
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        if '/bursa-dan-valas/' in href and 'd-' in href:
            if href.startswith('/'):
                href = "https://finance.detik.com" + href
            title = link.get_text(strip=True)
            if title and href not in [a['url'] for a in article_links]:
                article_links.append({'url': href, 'title': title})

    print(f"Menemukan {len(article_links)} artikel")

    all_articles = []
    for idx, link_info in enumerate(article_links[:max_articles], 1):
        print(f"[{idx}/{max_articles}] Scraping: {link_info['title'][:40]}...")
        df = scrape_detik(link_info['url'])
        if df is not None:
            all_articles.append(df)
        time.sleep(1)  # Hindari block

    if all_articles:
        final_df = pd.concat(all_articles, ignore_index=True)
        final_df.to_csv("artikel_detik_multiple.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ {len(final_df)} artikel disimpan ke artikel_detik_multiple.csv")
        return final_df
    return None

df = scrape_multiple_articles()
if df is None:
    print("\n⚠️ Fallback ke single article...")
    URL = "https://finance.detik.com/bursa-dan-valas/d-8517612/purbaya-buka-suara-dolar-as-tembus-rp-18-000-saya-serahkan-ke-bi"
    df = scrape_detik(URL)

if df is not None:
    df.to_csv(
        "artikel_detik.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(df.head())
    print(
        "\nArtikel berhasil disimpan ke artikel_detik.csv"
    )
else:
    print("Proses scraping gagal.")