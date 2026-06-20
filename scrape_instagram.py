import os
import re
import time
import warnings
import pandas as pd
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline, logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv

POST_URL = "https://www.instagram.com/reels/DZKBO1RoKIf/"
MAX_SCROLL = 30

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
load_dotenv()

IG_USER = os.getenv("IG_USER")
IG_PASS = os.getenv("IG_PASS")
HF_TOKEN = os.getenv("HF_TOKEN")

if not IG_USER or not IG_PASS:
    raise Exception("IG_USER atau IG_PASS belum ada di .env")

device = 0 if torch.cuda.is_available() else -1
print("CUDA Available:", torch.cuda.is_available())
print("Loading IndoBERT Sentiment Model...")

id_model = pipeline(
    "text-classification",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
    token=HF_TOKEN,
    device=device
)

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}


def clean_comment(text):
    text = text.strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)

    blacklist = [
        "view all", "reply", "replies", "see translation",
        "follow", "following", "original audio", "liked by",
        "log in", "sign up", "add a comment", "meta",
        "instagram", "threads", "verified", "search",
        "explore", "like", "likes"
    ]

    for word in blacklist:
        if word in text:
            return None

    if text.isdigit():
        return None
    if len(text.split()) < 3:
        return None
    if len(text) < 8:
        return None
    if len(text.split()) > 50:
        return None

    return text


def login_instagram(driver):
    wait = WebDriverWait(driver, 20)
    print("Membuka Instagram Login")

    driver.get("https://www.instagram.com/accounts/login/")
    time.sleep(5)

    username_input = wait.until(
        EC.presence_of_element_located((By.NAME, "email"))
    )
    password_input = wait.until(
        EC.presence_of_element_located((By.NAME, "pass"))
    )

    username_input.send_keys(IG_USER)
    password_input.send_keys(IG_PASS)
    password_input.send_keys(Keys.RETURN)
    print("Login berhasil")
    time.sleep(10)

    try:
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            txt = btn.text.lower()
            if "not now" in txt or "nanti saja" in txt:
                btn.click()
                time.sleep(2)
    except:
        pass


def scrape_comments(driver, post_url):
    print(f"\nMembuka postingan:")
    print(post_url)
    driver.get(post_url)
    time.sleep(8)

    for i in range(MAX_SCROLL):
        print(f"Scroll komentar {i + 1}")
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    comments = []
    spans = soup.find_all("span")

    for span in spans:
        try:
            text = span.get_text(strip=True)
            cleaned = clean_comment(text)
            if cleaned:
                comments.append(cleaned)
        except:
            pass

    comments = list(set(comments))
    print(f"\nKomentar valid ditemukan: {len(comments)}")
    return comments


def analyze_sentiment(comment):
    try:
        result = id_model(comment)[0]
        sentiment = label_map.get(result["label"], result["label"])
        confidence = round(result["score"], 4)
        return {
            "original_comment": comment,
            "sentiment": sentiment,
            "confidence": confidence
        }
    except Exception as e:
        print("Error sentiment:", e)
        return None


def analyze_narrative_gap(media_keywords, public_keywords):
    media_set = set(media_keywords)
    public_set = set(public_keywords)

    only_in_media = media_set - public_set
    only_in_public = public_set - media_set
    common = media_set & public_set

    return {
        'only_in_media': list(only_in_media),
        'only_in_public': list(only_in_public),
        'common_topics': list(common),
        'gap_ratio': len(only_in_public) / (len(common) + 1) if common else 1.0
    }


def extract_keywords_from_comments(comments, top_n=10):
    from collections import Counter

    all_text = " ".join(comments)
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

    stopwords = {'yang', 'dan', 'atau', 'untuk', 'dengan', 'dari', 'pada', 'dalam',
                 'karena', 'juga', 'akan', 'sudah', 'telah', 'masih', 'jadi', 'agar',
                 'bahwa', 'ini', 'itu', 'saat', 'lebih', 'saya', 'kamu', 'dia', 'kami',
                 'mereka', 'aku', 'kau', 'anda', 'pak', 'bu', 'mas', 'mbak', 'bro',
                 'sist', 'gan', 'bang', 'kak', 'om', 'tante'}

    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    counter = Counter(filtered_words)
    return counter.most_common(top_n)

def generate_insights(sentiment_distribution, gap_analysis, comment_count):
    insights = []

    if sentiment_distribution is not None and not sentiment_distribution.empty:
        dominant = sentiment_distribution.idxmax()

        if dominant == 'positive':
            insights.append({
                'title': 'Opini Publik Cenderung Positif',
                'description': f'{sentiment_distribution[dominant]:.1f}% komentar menunjukkan sentimen positif terhadap isu dolar',
                'severity': 'LOW'
            })
        elif dominant == 'negative':
            insights.append({
                'title': 'Kekhawatiran Publik Tinggi',
                'description': f'{sentiment_distribution[dominant]:.1f}% komentar menunjukkan sentimen negatif, publik khawatir dengan dampak dolar',
                'severity': 'HIGH'
            })
        else:
            insights.append({
                'title': 'Publik Bersikap Netral',
                'description': f'{sentiment_distribution[dominant]:.1f}% komentar netral, belum ada sikap jelas dari publik',
                'severity': 'MEDIUM'
            })
    else:
        insights.append({
            'title': 'Data Sentimen Tidak Tersedia',
            'description': 'Tidak cukup data untuk analisis sentimen',
            'severity': 'LOW'
        })

    # Insight 2: Gap analysis
    if gap_analysis and gap_analysis.get('only_in_public'):
        insights.append({
            'title': 'Isu Prioritas Publik yang Perlu Diperhatikan',
            'description': f'Publik banyak membahas: {", ".join(gap_analysis["only_in_public"][:3])}',
            'severity': 'HIGH'
        })

    # Insight 3: Volume komentar
    if comment_count > 100:
        insights.append({
            'title': 'Tingkat Engagement Tinggi',
            'description': f'Terdapat {comment_count} komentar, menunjukkan tingginya perhatian publik terhadap isu ini',
            'severity': 'MEDIUM'
        })
    elif comment_count > 50:
        insights.append({
            'title': 'Perhatian Publik Sedang',
            'description': f'{comment_count} komentar terkumpul, publik cukup memperhatikan isu dolar',
            'severity': 'LOW'
        })
    else:
        insights.append({
            'title': 'Perhatian Publik Rendah',
            'description': f'Hanya {comment_count} komentar, publik belum terlalu peduli dengan isu ini',
            'severity': 'LOW'
        })

    while len(insights) < 3:
        insights.append({
            'title': f'Insight {len(insights) + 1}',
            'description': 'Perlu monitoring lanjutan untuk identifikasi pola tambahan',
            'severity': 'MEDIUM'
        })

    return insights

options = webdriver.ChromeOptions()
options.add_argument("--disable-notifications")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

driver = webdriver.Chrome(options=options)

try:
    login_instagram(driver)

    print("\n" + "=" * 70)
    print("SCRAPING INSTAGRAM REELS")
    print("=" * 70)

    comments = scrape_comments(driver, POST_URL)

    all_results = []
    all_comments = comments

    for comment in comments:
        result = analyze_sentiment(comment)
        if result:
            result["post_url"] = POST_URL
            all_results.append(result)
            print("\nTEXT :", comment)
            print("SENTIMENT :", result["sentiment"], result["confidence"])

finally:
    driver.quit()

df = pd.DataFrame(all_results)
df.to_csv("instagram_sentiment.csv", index=False, encoding="utf-8")

comments_df = pd.DataFrame({
    "comment": all_comments,
    "source": ["instagram"] * len(all_comments)
})
comments_df.to_csv("instagram_comments.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 50)
print("ANALISIS KEYWORD PUBLIK")
print("=" * 50)

top_keywords = extract_keywords_from_comments(all_comments, 10)
print("\nTop 10 Keyword Publik:")
for kw, freq in top_keywords:
    print(f"  {kw}: {freq}")

keyword_df = pd.DataFrame(top_keywords, columns=["keyword", "frequency"])
keyword_df.to_csv("public_keywords.csv", index=False, encoding="utf-8-sig")
print("\n✅ public_keywords.csv berhasil dibuat")

gap_analysis = None
try:
    media_df = pd.read_csv("media_keywords.csv")
    media_keywords = media_df['keyword'].tolist()
    public_keywords = [kw for kw, _ in top_keywords]

    gap_analysis = analyze_narrative_gap(media_keywords, public_keywords)

    print("\n" + "=" * 50)
    print("ANALISIS KESENJANGAN NARASI")
    print("=" * 50)
    print(f"🔍 Topik hanya di Media: {gap_analysis['only_in_media'][:5]}")
    print(f"🔍 Topik hanya di Publik: {gap_analysis['only_in_public'][:5]}")
    print(f"🔍 Topik yang sama: {gap_analysis['common_topics'][:5]}")
    print(f"📊 Gap Ratio: {gap_analysis['gap_ratio']:.2f}")

    gap_df = pd.DataFrame({
        'only_in_media': pd.Series(gap_analysis['only_in_media']),
        'only_in_public': pd.Series(gap_analysis['only_in_public']),
        'common_topics': pd.Series(gap_analysis['common_topics'])
    })
    gap_df.to_csv("narrative_gap_analysis.csv", index=False, encoding="utf-8-sig")
    print("\n✅ narrative_gap_analysis.csv berhasil dibuat")

except FileNotFoundError:
    print("\n⚠️ File media_keywords.csv tidak ditemukan, gap analysis dilewati")
    gap_analysis = {'only_in_public': []}

# ===== INSIGHTS =====
print("\n" + "=" * 50)
print("3 INSIGHT KRITIS")
print("=" * 50)

sentiment_dist = df['sentiment'].value_counts()
insights = generate_insights(sentiment_dist, gap_analysis if gap_analysis else {'only_in_public': []}, len(df))

for i, ins in enumerate(insights, 1):
    print(f"\n{i}. [{ins['severity']}] {ins['title']}")
    print(f"   {ins['description']}")

print("\n" + "=" * 50)
print("SELESAI")
print("=" * 50)
print(f"Total data: {len(df)}")
print("Distribusi Sentimen:")
print(df["sentiment"].value_counts())
print("\nFile yang dihasilkan:")
print("📁 instagram_sentiment.csv")
print("📁 instagram_comments.csv")
print("📁 public_keywords.csv")
if gap_analysis and gap_analysis.get('only_in_public'):
    print("📁 narrative_gap_analysis.csv")