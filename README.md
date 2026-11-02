# Sentube ⚡
### Large-Scale Comment Sentiment Mining and Audience Feedback Analytics Using PySpark and YouTube Data API v3

> **BDSMA Academic Micro-Project**  
> Distributed Data Processing • Real-Time Ingestion • VADER Sentiment UDF • Spam Bot Telemetry • Audience Discrepancy Index

---

## 📖 Overview
**Sentube** is an automated big data analytics system engineered to evaluate genuine audience reception on YouTube at scale. By querying the YouTube Data API v3, Sentube ingests thousands of public comments and partitions the workload across a distributed **PySpark** cluster. 

The engine strips noise (emojis, URLs, punctuation), maps sentiment polarity via a VADER User-Defined Function (UDF), flags commercial bot spam, extracts key recurring bigrams (e.g. *"audio quality"*, *"explained well"*, *"price hike"*), and calculates the **Audience Discrepancy Index**—detecting videos with artificially inflated like counts or misleading thumbnails that harbor hostile comments.

---

## 🏛️ System Architecture

```
[ YouTube Data API v3 ]
          │ (JSON Response / Pagination)
          ▼
[ Data Ingestion / CSV Staging ]
          │
          ▼
[ PySpark Distributed Engine ] 
   ├── Text Cleaning & Tokenization (Regex, StopWordsRemover)
   ├── Sentiment Polarity Scoring (VADER UDF)
   ├── Topic & Keyword Extraction (TF-IDF / Bigrams)
   └── Spam & Self-Promotion Rule Filter (WhatsApp, Telegram, Crypto)
          │
          ▼
[ Streamlit Web Telemetry Dashboard ]
   ├── Real-Time Polarity Donut (Positive / Neutral / Negative)
   ├── Bigram Topic Frequency Bar Chart
   ├── Spam vs Verified Audience Telemetry Gauge
   └── Audience Discrepancy Index Meter
```

---

## 🚀 Key Features

1. **Distributed Sentiment & Polarity Scoring (PySpark UDF):**
   - Distributed regex transformations clean emojis and URLs.
   - Maps comments into Positive, Neutral, or Negative polarity using NLTK VADER compound scores inside a parallel PySpark UDF.
2. **Spam & Self-Promotion Rule Engine:**
   - Detects financial fraud bots, WhatsApp phone patterns, Telegram pump channels, and channel self-promotion strings.
   - Computes an instant Clean-to-Spam ratio.
3. **Topic & Keyword Highlights (TF-IDF Bigrams):**
   - Automatically isolates the most recurring two-word phrases to reveal the exact root causes of viewer praise or criticism.
4. **Audience Discrepancy Index:**
   - Quantifies the divergence between the video's public Like-to-View ratio and the actual comment sentiment distribution.
   - Immediately isolates clickbait videos or updates with community backlash.
5. **Interactive Streamlit Web Dashboard:**
   - Single-page application designed with a **Dark Mode First & Tech Forward** kinetic aesthetic.
   - Includes one-click academic presets for zero-friction evaluation.

---

## 🛠️ Tech Stack & Academic Justification

| Layer | Tool / Library | BDSMA Academic Justification |
| :--- | :--- | :--- |
| **Data Ingestion** | `google-api-python-client` | Free YouTube Data API v3 tier (10,000 units/day; 100 comments cost only 1 quota unit). |
| **Big Data Engine** | `pyspark` (SQL & MLlib) | Satisfies university distributed parallel processing & DataFrame UDF criteria. |
| **NLP** | `nltk` (VADER) | Fast, rule-based lexicon scoring requiring zero GPU overhead. |
| **Dashboard** | `streamlit` + `plotly` | High-impact interactive visualization with zero complex JavaScript setup. |

---

## 💻 Quick Start & Evaluation Guide

### 1. Installation
Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```

The application will launch automatically at `http://localhost:8501`.

### 3. Examiner Demo Modes
- **Zero-Config Pre-staged Mode (Default):** Select any of the 3 built-in demo scenarios from the sidebar (*Controversial Keynote*, *Beloved Tutorial*, or *Crypto Bot Trap*) to immediately test the distributed pipeline without needing an active YouTube API key.
- **Live YouTube Mode:** Enter your YouTube Data API v3 key in the sidebar, paste any YouTube URL or video ID, and click **"Process with PySpark"**.

---

## 📁 Repository Structure
```
BDSMA-MicroProject/
├── .streamlit/
│   └── config.toml             # Dark mode theme & layout configuration
├── sample_data/
│   ├── controversial_keynote.csv # High discrepancy test scenario
│   ├── tutorial_loved.csv       # High positive test scenario
│   └── crypto_spam.csv          # High spam ratio test scenario
├── app.py                      # Streamlit interactive landing page & dashboard
├── spark_engine.py             # PySpark transformations, UDFs, and discrepancy metrics
├── youtube_api.py              # YouTube Data API v3 ingestion & fallback handler
├── index.html                  # Standalone modern web landing page
├── requirements.txt            # Pinned dependency definitions
└── README.md                   # Project documentation & evaluation criteria
```
