import re
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from transformers import pipeline

print("Loading IndoBERT Model...")

model = pipeline(
    "text-classification",
    model="w11wo/indonesian-roberta-base-sentiment-classifier"
)

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def predict_sentiment(text):
    try:
        result = model(str(text)[:512])[0]
        label = result["label"]
        if label in label_map:
            sentiment = label_map[label]
        else:
            sentiment = str(label).lower()
        return {
            "sentiment": sentiment,
            "score": round(result["score"], 4)
        }
    except Exception as e:
        print("Error:", e)
        return {
            "sentiment": "unknown",
            "score": 0
        }

factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

custom_stopwords = {
    "yang", "dan", "atau", "untuk", "dengan",
    "dari", "pada", "dalam", "karena",
    "juga", "akan", "sudah", "telah",
    "masih", "jadi", "agar", "bahwa",
    "ini", "itu", "saat", "lebih",
    "detik", "com", "id",
    "instagram",
    "pak",
    "purbaya",
    "rp",
    "000"
}

stopwords = stopwords.union(
    custom_stopwords
)


# ===== FUNGSI ORIGINAL (TIDAK DIUBAH) =====
def clean_text(text):
    words = re.findall(
        r'\b[a-zA-Z]+\b',
        str(text).lower()
    )
    return [
        word
        for word in words
        if word not in stopwords
           and len(word) > 2
    ]

def get_top_keywords(text, n=10):
    words = clean_text(text)
    counter = Counter(words)
    return pd.DataFrame(
        counter.most_common(n),
        columns=[
            "keyword",
            "frequency"
        ]
    )


# ===== TAMBAHAN: FUNGSI GAP ANALYSIS =====
def narrative_gap_analysis(media_keywords, public_keywords, top_n=10):
    """
    Analisis kesenjangan narasi antara media dan publik
    """
    media_set = set(media_keywords['keyword'].head(top_n))
    public_set = set(public_keywords['keyword'].head(top_n))

    only_in_media = media_set - public_set
    only_in_public = public_set - media_set
    common = media_set & public_set

    return {
        'only_in_media': list(only_in_media),
        'only_in_public': list(only_in_public),
        'common_topics': list(common),
        'gap_ratio': len(only_in_public) / (len(common) + 1) if common else 1.0
    }

def generate_insights(media_sentiment, public_sentiment, gap_analysis):
    insights = []

    # Insight 1: Sentimen Gap
    if media_sentiment != public_sentiment:
        severity = 'HIGH' if (media_sentiment == 'positive' and public_sentiment == 'negative') else 'MEDIUM'
        insights.append({
            'title': 'Kesenjangan Sentimen',
            'description': f'Media cenderung {media_sentiment}, publik cenderung {public_sentiment}',
            'severity': severity
        })

    # Insight 2: Topik yang tidak tercover
    if gap_analysis['only_in_public']:
        insights.append({
            'title': 'Isu Prioritas Publik yang Tidak Tercover Media',
            'description': f'Publik membahas: {", ".join(gap_analysis["only_in_public"][:3])}',
            'severity': 'HIGH'
        })

    # Insight 3: Gap Ratio
    if gap_analysis['gap_ratio'] > 0.5:
        insights.append({
            'title': 'Kesenjangan Narasi Signifikan',
            'description': f'Gap ratio {gap_analysis["gap_ratio"]:.2f} menunjukkan perbedaan fokus yang besar',
            'severity': 'HIGH'
        })

    # Jika kurang dari 3, tambahkan default
    while len(insights) < 3:
        insights.append({
            'title': f'Insight {len(insights) + 1}',
            'description': 'Perlu monitoring lanjutan untuk identifikasi pola tambahan',
            'severity': 'MEDIUM'
        })

    return insights

def create_visualization(sentiment_data, media_kw, public_kw, gap_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analisis Sentimen & Kesenjangan Narasi', fontsize=16, fontweight='bold')

    # 1. Sentimen Comparison
    ax1 = axes[0, 0]
    sentiment_data.plot(kind='bar', ax=ax1, rot=0)
    ax1.set_title('Perbandingan Sentimen')
    ax1.set_ylabel('Persentase (%)')
    ax1.legend(title='Sentimen')
    ax1.grid(True, alpha=0.3)

    # 2. Keyword Comparison
    ax2 = axes[0, 1]
    media_top = media_kw.head(5)['keyword'].tolist()
    public_top = public_kw.head(5)['keyword'].tolist()

    all_keywords = list(set(media_top + public_top))
    media_present = [1 if k in media_top else 0 for k in all_keywords]
    public_present = [1 if k in public_top else 0 for k in all_keywords]

    x = np.arange(len(all_keywords))
    width = 0.35
    ax2.bar(x - width / 2, media_present, width, label='Media', alpha=0.8)
    ax2.bar(x + width / 2, public_present, width, label='Publik', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_keywords, rotation=45, ha='right')
    ax2.set_title('Top 5 Keyword: Media vs Publik')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Gap Analysis
    ax3 = axes[1, 0]
    gap_data_plot = pd.DataFrame({
        'Kategori': ['Hanya Media', 'Hanya Publik', 'Sama'],
        'Jumlah': [
            len(gap_data['only_in_media']),
            len(gap_data['only_in_public']),
            len(gap_data['common_topics'])
        ]
    })
    ax3.bar(gap_data_plot['Kategori'], gap_data_plot['Jumlah'], color=['blue', 'red', 'green'])
    ax3.set_title('Analisis Kesenjangan Narasi')
    ax3.set_ylabel('Jumlah Topik')
    ax3.grid(True, alpha=0.3)

    # 4. Insights Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Dapatkan sentimen dominan
    media_sentiment = sentiment_data['Media'].idxmax() if not sentiment_data['Media'].empty else 'unknown'
    public_sentiment = sentiment_data['Publik'].idxmax() if not sentiment_data['Publik'].empty else 'unknown'

    insights = generate_insights(media_sentiment, public_sentiment, gap_data)

    text = "3 INSIGHT KRITIS:\n\n"
    for i, ins in enumerate(insights, 1):
        text += f"{i}. [{ins['severity']}] {ins['title']}\n"
        text += f"   {ins['description']}\n\n"
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig('sentiment_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualisasi disimpan: sentiment_visualization.png")
    plt.show()

print("Loading dataset...")
try:
    article_df = pd.read_csv("artikel_detik_multiple.csv")
    print(f"✅ Memuat {len(article_df)} artikel dari artikel_detik_multiple.csv")
except FileNotFoundError:
    try:
        article_df = pd.read_csv("artikel_detik.csv")
        print(f"✅ Memuat 1 artikel dari artikel_detik.csv")
    except FileNotFoundError:
        print("❌ File artikel tidak ditemukan!")
        exit(1)

# Coba load multiple comments dulu
try:
    comments_df = pd.read_csv("instagram_comments_multiple.csv")
    print(f"✅ Memuat {len(comments_df)} komentar dari instagram_comments_multiple.csv")
except FileNotFoundError:
    try:
        comments_df = pd.read_csv("instagram_comments.csv")
        print(f"✅ Memuat {len(comments_df)} komentar dari instagram_comments.csv")
    except FileNotFoundError:
        print("❌ File komentar tidak ditemukan!")
        exit(1)

comments_df = comments_df.dropna(subset=["comment"])
article_text = article_df.loc[
    0,
    "content"
]

sentences = re.split(
    r"[.!?]",
    str(article_text)
)

media_results = []

print("\nAnalisis Sentimen Artikel...")

for sentence in sentences:
    sentence = sentence.strip()
    if len(sentence) < 15:
        continue
    result = predict_sentiment(sentence)
    media_results.append({
        "source": "media",
        "text": sentence,
        "sentiment": result["sentiment"],
        "score": result["score"]
    })

public_results = []

print("Analisis Sentimen Komentar Instagram...")

for comment in comments_df["comment"]:
    result = predict_sentiment(comment)
    public_results.append({
        "source": "instagram",
        "text": comment,
        "sentiment": result["sentiment"],
        "score": result["score"]
    })

final_df = pd.DataFrame(
    media_results + public_results
)

final_df.to_csv(
    "final_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nfinal_result.csv berhasil dibuat")
media_summary = (
        final_df[
            final_df["source"] == "media"
            ]["sentiment"]
        .value_counts(normalize=True)
        * 100
)

print("\n==========================")
print("SENTIMEN MEDIA (%)")
print("==========================")
print(
    media_summary.round(2)
)

public_summary = (
        final_df[
            final_df["source"] == "instagram"
            ]["sentiment"]
        .value_counts(normalize=True)
        * 100
)

print("\n==========================")
print("SENTIMEN PUBLIK (%)")
print("==========================")
print(
    public_summary.round(2)
)

media_dominant = media_summary.idxmax()
public_dominant = public_summary.idxmax()

print("\n==========================")
print("RINGKASAN")
print("==========================")

print(
    f"Sentimen dominan media  : {media_dominant}"
)
print(
    f"Sentimen dominan publik : {public_dominant}"
)

if media_dominant != public_dominant:
    print(
        "\nTerdapat kesenjangan narasi antara media dan publik."
    )
else:
    print(
        "\nNarasi media relatif sejalan dengan opini publik."
    )

summary_df = pd.DataFrame({
    "source": [
        "media",
        "instagram"
    ],
    "dominant_sentiment": [
        media_dominant,
        public_dominant
    ]
})

summary_df.to_csv(
    "sentiment_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nsentiment_summary.csv berhasil dibuat"
)
print(
    "\nAnalisis Keyword Media..."
)

media_keywords = get_top_keywords(
    article_text,
    10
)

media_keywords.to_csv(
    "media_keywords.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop Keyword Media")
print(media_keywords)

print(
    "\nAnalisis Keyword Publik..."
)

public_text = " ".join(
    comments_df["comment"].astype(str)
)

public_keywords = get_top_keywords(
    public_text,
    10
)

public_keywords.to_csv(
    "public_keywords.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop Keyword Publik")
print(public_keywords)

print(
    "\nmedia_keywords.csv berhasil dibuat"
)
print(
    "public_keywords.csv berhasil dibuat"
)

print("\n" + "=" * 50)
print("ANALISIS KESENJANGAN NARASI")
print("=" * 50)

gap_analysis = narrative_gap_analysis(media_keywords, public_keywords)

print(f"\n🔍 Topik hanya di Media: {gap_analysis['only_in_media'][:5]}")
print(f"🔍 Topik hanya di Publik: {gap_analysis['only_in_public'][:5]}")
print(f"🔍 Topik yang sama: {gap_analysis['common_topics'][:5]}")
print(f"📊 Gap Ratio: {gap_analysis['gap_ratio']:.2f}")

# Simpan gap analysis
gap_df = pd.DataFrame({
    'only_in_media': pd.Series(gap_analysis['only_in_media']),
    'only_in_public': pd.Series(gap_analysis['only_in_public']),
    'common_topics': pd.Series(gap_analysis['common_topics'])
})
gap_df.to_csv("narrative_gap_analysis.csv", index=False, encoding="utf-8-sig")
print("\n✅ narrative_gap_analysis.csv berhasil dibuat")

# ===== TAMBAHAN: INSIGHT GENERATION =====
insights = generate_insights(media_dominant, public_dominant, gap_analysis)

print("\n" + "=" * 50)
print("3 INSIGHT KRITIS")
print("=" * 50)
for i, ins in enumerate(insights, 1):
    print(f"\n{i}. [{ins['severity']}] {ins['title']}")
    print(f"   {ins['description']}")

print("\n📊 Membuat visualisasi...")

sentiment_data = pd.DataFrame({
    'Media': media_summary,
    'Publik': public_summary
}).fillna(0)

create_visualization(sentiment_data, media_keywords, public_keywords, gap_analysis)

print("\n" + "=" * 50)
print("RINGKASAN FINAL")
print("=" * 50)
print(f"📅 Analisis selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📈 Sentimen Dominan Media: {media_dominant}")
print(f"📈 Sentimen Dominan Publik: {public_dominant}")

if media_dominant != public_dominant:
    print("\n⚠️ TERDAPAT KESENJANGAN NARASI!")
    print(f"   Media cenderung {media_dominant}, publik cenderung {public_dominant}")
else:
    print("\n✅ Narasi media relatif sejalan dengan opini publik")

print("\n📁 File yang dihasilkan:")
print("   - artikel_detik.csv / artikel_detik_multiple.csv")
print("   - instagram_comments.csv / instagram_comments_multiple.csv")
print("   - final_result.csv")
print("   - sentiment_summary.csv")
print("   - narrative_gap_analysis.csv")
print("   - sentiment_visualization.png")
print("   - media_keywords.csv, public_keywords.csv")

print("\nSample Hasil")
print(
    final_df.head(10)
)

print("\nANALISIS SELESAI")