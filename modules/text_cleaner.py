import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only runs once)
NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "punkt": "tokenizers/punkt",
}

for pkg, path in NLTK_PACKAGES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def remove_html(text: str) -> str:
    text = html.unescape(text)                    # &amp; -> &,  â€™ -> '
    text = re.sub(r"<[^>]+>", " ", text)          # <div class=...> -> space
    text = re.sub(r"&[a-zA-Z]+;", " ", text)      # leftover entities
    return text


def fix_encoding(text: str) -> str:
    replacements = {
        "â€™": "'", "â€œ": '"', "â€": '"',
        "â€¢": "-", "Â": "", "â€": "-",
        "â€˜": "'", "\u00e2\u0080\u0099": "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_noise(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)           # URLs
    text = re.sub(r"\S+@\S+", " ", text)                     # emails
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", " ", text)  # phones
    text = re.sub(r"[^a-z\s]", " ", text)                    # keep only letters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_filter(text: str) -> list[str]:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def clean_text(text: str, return_tokens: bool = False):
    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""

    text = fix_encoding(text)
    text = remove_html(text)
    text = normalize(text)
    text = remove_noise(text)
    tokens = tokenize_and_filter(text)

    return tokens if return_tokens else " ".join(tokens)


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    <div class="fontsize fontface">HR SPECIALIST â€" Summary<br/>
    10+ years experience in Human Resources. Skilled in recruitment,
    performance management, and employee relations.
    Contact: john@email.com | 555-123-4567
    </div>
    """
    print("RAW:\n", sample[:120])
    print("\nCLEANED (string):\n", clean_text(sample))
    print("\nCLEANED (tokens):\n", clean_text(sample, return_tokens=True))
