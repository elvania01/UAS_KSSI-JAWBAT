# Analisis Sentimen & Kesenjangan Narasi: Media vs Publik

## Kasus: Dolar AS Tembus Rp 18.000

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis **kesenjangan narasi (narrative gap)** antara media massa dan opini publik terkait isu **Dolar AS Tembus Rp 18.000**.

Media yang digunakan sebagai sumber data adalah artikel berita dari **Detik.com**, sedangkan opini publik diperoleh dari **komentar pengguna Instagram** pada unggahan yang membahas isu terkait.

Analisis dilakukan melalui beberapa tahapan:

* **Web Scraping** artikel berita dari Detik.com
* **Scraping Komentar Instagram** menggunakan Instaloader
* **Sentiment Analysis** menggunakan model IndoBERT
* **Keyword Extraction** untuk mengidentifikasi topik dominan
* **Narrative Gap Analysis** untuk membandingkan fokus pembahasan media dan publik

---

## 🎯 Tujuan Proyek

1. Mengetahui apakah narasi media sinkron dengan opini publik.
2. Mengidentifikasi perbedaan fokus pembahasan antara media dan masyarakat.
3. Menemukan potensi kesenjangan narasi yang dapat memengaruhi persepsi publik.
4. Memberikan rekomendasi strategis bagi pengambil kebijakan dan media.

---

## 📊 Dataset

### Sumber Data Media

* Artikel berita dari Detik.com terkait isu nilai tukar Rupiah terhadap Dolar AS.

### Sumber Data Publik

* Komentar pengguna Instagram pada unggahan yang membahas kenaikan nilai tukar Dolar AS.

---

## 📦 Dependencies

Library yang digunakan dalam proyek ini:

* pandas
* numpy
* matplotlib
* seaborn
* transformers
* torch
* scikit-learn
* instaloader
* beautifulsoup4
* requests
* python-dotenv

Instal seluruh dependency menggunakan:

```bash
pip install -r requirements.txt
```

---

## 📁 Struktur Proyek

```text
UAS/
│
├── .env
├── requirements.txt
├── README.md
│
├── scrape_berita.py
├── scrape_instagram.py
├── analisis.py
│
├── hasil_scraping/
│   ├── artikel_detik.csv
│   ├── instagram_comments.csv
│   ├── final_result.csv
│   ├── sentiment_summary.csv
│   ├── media_keywords.csv
│   ├── public_keywords.csv
│   ├── narrative_gap_analysis.csv
│   └── sentiment_visualization.png
│
└── docs/
    └── presentasi.pptx
```

### Keterangan File

| File                        | Fungsi                                                 |
| --------------------------- | ------------------------------------------------------ |
| scrape_berita.py            | Mengambil artikel berita dari Detik.com                |
| scrape_instagram.py         | Mengambil komentar Instagram                           |
| analisis.py                 | Analisis sentimen, ekstraksi keyword, dan gap analysis |
| artikel_detik.csv           | Hasil scraping berita                                  |
| instagram_comments.csv      | Hasil scraping komentar                                |
| final_result.csv            | Dataset hasil pengolahan                               |
| sentiment_summary.csv       | Ringkasan sentimen                                     |
| media_keywords.csv          | Keyword dominan media                                  |
| public_keywords.csv         | Keyword dominan publik                                 |
| narrative_gap_analysis.csv  | Hasil analisis kesenjangan narasi                      |
| sentiment_visualization.png | Visualisasi sentimen                                   |

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/UAS-KSSI.git

cd UAS-KSSI
```

---

### 2. Buat Virtual Environment (Opsional)

#### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Konfigurasi Instagram

Buat file `.env` pada root project:

```env
IG_USERNAME=your_username
IG_PASSWORD=your_password
```

Ganti `your_username` dan `your_password` dengan akun Instagram yang digunakan untuk proses scraping.

---

### 5. Jalankan Scraping Artikel Berita

```bash
python scrape_berita.py
```

Output:

```text
hasil_scraping/artikel_detik.csv
```

---

### 6. Jalankan Scraping Komentar Instagram

```bash
python scrape_instagram.py
```

Output:

```text
hasil_scraping/instagram_comments.csv
```

---

### 7. Jalankan Analisis Sentimen dan Gap Analysis

```bash
python analisis.py
```

Output:

```text
hasil_scraping/final_result.csv
hasil_scraping/sentiment_summary.csv
hasil_scraping/media_keywords.csv
hasil_scraping/public_keywords.csv
hasil_scraping/narrative_gap_analysis.csv
hasil_scraping/sentiment_visualization.png
```

---

## 📤 Output yang Dihasilkan

| File Output                 | Deskripsi                       |
| --------------------------- | ------------------------------- |
| artikel_detik.csv           | Data artikel hasil scraping     |
| instagram_comments.csv      | Data komentar Instagram         |
| final_result.csv            | Dataset gabungan untuk analisis |
| sentiment_summary.csv       | Ringkasan hasil sentimen        |
| media_keywords.csv          | Topik dominan media             |
| public_keywords.csv         | Topik dominan publik            |
| narrative_gap_analysis.csv  | Analisis kesenjangan narasi     |
| sentiment_visualization.png | Grafik visualisasi sentimen     |

---

## 📈 Visualisasi Sentimen

Visualisasi distribusi sentimen akan disimpan pada:

```text
hasil_scraping/sentiment_visualization.png
```

Apabila file tersedia di repository, gambar dapat ditampilkan langsung menggunakan:

```markdown
![Visualisasi Sentimen](hasil_scraping/sentiment_visualization.png)
```

---

## 📌 Ringkasan Hasil

### Narasi Media

* Dominan memiliki sentimen netral.
* Fokus pada stabilitas ekonomi dan indikator makro.
* Menyoroti kebijakan pemerintah dan kondisi pasar.

### Narasi Publik

* Dominan memiliki sentimen negatif.
* Fokus pada kenaikan harga kebutuhan pokok.
* Menyoroti dampak langsung terhadap kehidupan sehari-hari.

### Kesenjangan Narasi

Ditemukan adanya perbedaan fokus pembahasan antara media dan masyarakat:

| Media                   | Publik                                |
| ----------------------- | ------------------------------------- |
| Stabilitas ekonomi      | Harga kebutuhan pokok                 |
| Kurs dan pasar keuangan | Daya beli masyarakat                  |
| Kebijakan pemerintah    | Dampak terhadap kehidupan sehari-hari |

Perbedaan ini menunjukkan bahwa media lebih banyak membahas aspek makroekonomi, sementara masyarakat lebih memperhatikan dampak ekonomi yang dirasakan secara langsung.

---

## 💡 Rekomendasi

1. Media perlu lebih banyak mengangkat dampak ekonomi pada masyarakat.
2. Pemerintah perlu memperkuat komunikasi publik terkait kondisi ekonomi.
3. Pemantauan sentimen publik dapat digunakan sebagai indikator respons masyarakat terhadap isu ekonomi.

---

## 👨‍💻 Author

Kelompok UAS Kecerdasan Sistem dan Sentimen Informasi (KSSI)

* Elvania Pranisti (231712660)
* Iin Anugrah Sinambela (231712621)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik dan tugas Ujian Akhir Semester (UAS).
