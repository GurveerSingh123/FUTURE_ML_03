# Resume Screening System

An NLP-based resume screening and candidate ranking system built with
Python, scikit-learn, spaCy, and NLTK.

---

## Project Structure

```
resume_screener/
├── data/
│   └── Resume.csv              ← Place your dataset here
├── modules/
│   ├── text_cleaner.py         ← Module 1: Clean raw resume text
│   ├── skill_extractor.py      ← Module 2: Extract skills via dictionary + NER
│   ├── screener.py             ← Module 3: TF-IDF vectorizer + cosine scorer
│   └── reporter.py             ← Module 4: Ranked output + skill gap report
├── output/                     ← Reports saved here automatically
├── main.py                     ← Entry point — run this
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy English model
python -m spacy download en_core_web_sm

# 4. Place Resume.csv in the data/ folder

# 5. Run the pipeline
python main.py
```

---

## How It Works

### Step 1 — Text Cleaning (text_cleaner.py)
Every resume goes through a cleaning pipeline:
- HTML tags stripped
- Broken unicode fixed
- Lowercased, punctuation removed
- Stopwords removed (NLTK)
- Lemmatization applied (running → run)

### Step 2 — Skill Extraction (skill_extractor.py)
Skills are extracted via two layers:
1. **Dictionary match** — 150+ skills across 12 domains
2. **spaCy NER** — catches technology product names the dict misses

### Step 3 — Vectorization & Scoring (screener.py)
- All resumes vectorized with TF-IDF (10K features, unigrams + bigrams)
- Job description vectorized with same vocabulary
- Cosine similarity computed between JD and every resume
- **Composite score = 0.6 × cosine_similarity + 0.4 × skill_match_ratio**

### Step 4 — Report Generation (reporter.py)
For each top-N candidate:
- Final score + quality band (Excellent / Good / Fair / Weak)
- Matched skills (candidate has AND JD requires)
- Missing skills (JD requires but candidate lacks)
- Actionable recommendation (Shortlist / Consider / Review / Pass)

---

## Scoring Formula

| Component | Weight | What it measures |
|---|---|---|
| Cosine similarity | 60% | Overall language + context match |
| Skill match ratio | 40% | Required skills coverage |

Score bands:
- **Excellent** ≥ 0.70
- **Good**      ≥ 0.50
- **Fair**      ≥ 0.30
- **Weak**      < 0.30

---

## Customising for Your Own Job Description

Edit `main.py` and add your JD to the `JOB_DESCRIPTIONS` dict:

```python
JOB_DESCRIPTIONS = {
    "Your Job Title": """
    Paste full job description here...
    """,
}
```

Or use the screener directly in your own script:

```python
from modules.screener import ResumeScreener
import pandas as pd

df = pd.read_csv("data/Resume.csv")
screener = ResumeScreener()
screener.fit(df)

results = screener.screen("Your job description text here", top_n=10)
print(results)
```

---

## Output Files

Every run saves two files to `/output`:
- `report_<job>_<timestamp>.txt` — full human-readable report
- `results_<job>_<timestamp>.csv` — machine-readable ranked results

---

## Tools Used

| Tool | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| scikit-learn TfidfVectorizer | Text vectorization |
| scikit-learn cosine_similarity | Similarity scoring |
| spaCy en_core_web_sm | Named entity recognition |
| NLTK | Stopwords, tokenization, lemmatization |

---

## Dataset

**Resume.csv** from Kaggle:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

- 2100+ resumes across 24 job categories
- Fields used: `Resume_str` (text), `Category` (label), `ID`
