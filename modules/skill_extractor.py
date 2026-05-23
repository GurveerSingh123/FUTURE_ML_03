import re
import spacy
from modules.text_cleaner import clean_text, normalize

# Load spaCy model (small English model — fast and sufficient)
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   capture_output=True)
    NLP = spacy.load("en_core_web_sm")


# ── Skill Dictionary ─────────────────────────────────────────────────────────
# Organised by domain for maintainability.
# All entries are lowercase; multi-word phrases are included.

SKILL_DICT = {
    # ── Programming Languages ──
    "python", "java", "javascript", "typescript", "c++", "c#", "r",
    "scala", "go", "golang", "kotlin", "swift", "ruby", "php", "perl",
    "bash", "shell", "matlab", "julia",

    # ── Web & Frontend ──
    "html", "css", "react", "angular", "vue", "next.js", "node.js",
    "express", "django", "flask", "fastapi", "spring", "rest api",
    "graphql", "webpack", "bootstrap", "tailwind",

    # ── Data & ML ──
    "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "data science", "data analysis",
    "data visualization", "feature engineering", "model deployment",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "xgboost", "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "hugging face", "transformers", "bert", "gpt",

    # ── Databases ──
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis",
    "cassandra", "elasticsearch", "oracle", "sql server", "nosql",
    "database design", "query optimization",

    # ── Cloud & DevOps ──
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "ci/cd", "jenkins", "github actions",
    "linux", "git", "github", "devops", "microservices",

    # ── HR Domain ──
    "recruitment", "talent acquisition", "onboarding", "hris",
    "performance management", "employee relations", "payroll",
    "compensation", "benefits administration", "hr policy",
    "labor law", "organizational development", "training",
    "succession planning", "workforce planning", "hr analytics",
    "workday", "sap hr", "bamboohr", "applicant tracking system",

    # ── Finance & Accounting ──
    "financial modeling", "accounting", "budgeting", "forecasting",
    "excel", "financial analysis", "gaap", "ifrs", "auditing",
    "tax", "quickbooks", "sap", "erp",

    # ── Marketing ──
    "seo", "sem", "social media marketing", "content marketing",
    "email marketing", "google analytics", "crm", "salesforce",
    "hubspot", "brand management", "market research", "a/b testing",

    # ── Project Management ──
    "project management", "agile", "scrum", "kanban", "jira",
    "confluence", "pmp", "stakeholder management", "risk management",
    "waterfall", "product management", "roadmap",

    # ── Soft / General ──
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management", "presentation",
    "negotiation", "conflict resolution", "mentoring", "coaching",

    # ── Healthcare ──
    "patient care", "clinical", "ehr", "hipaa", "medical coding",
    "nursing", "pharmacy", "diagnosis", "treatment planning",

    # ── Legal ──
    "contract law", "litigation", "compliance", "legal research",
    "corporate law", "intellectual property", "due diligence",

    # ── Design ──
    "ui design", "ux design", "figma", "sketch", "adobe xd",
    "photoshop", "illustrator", "wireframing", "prototyping",
    "user research", "design thinking",
}

# Sort by length descending so longer phrases are matched first
# "machine learning" before "machine" and "learning"
SORTED_SKILLS = sorted(SKILL_DICT, key=len, reverse=True)

# Entities labelled by spaCy NER that are probably real skills/tools
NER_SKILL_LABELS = {"ORG", "PRODUCT", "WORK_OF_ART"}

# Non-skill org names to block from NER results
NER_BLOCKLIST = {
    "university", "college", "institute", "school", "hospital",
    "inc", "llc", "ltd", "corp", "company", "group", "department",
    "team", "committee", "board", "foundation", "association",
}


def extract_from_dictionary(text: str) -> set[str]:
    """
    Scan cleaned text for known skill phrases.
    Uses word-boundary regex to avoid 'r' matching inside 'hr'.
    """
    found = set()
    text = normalize(text)
    for skill in SORTED_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(skill)
    return found


def extract_from_ner(text: str) -> set[str]:
    """
    Use spaCy NER to find product/org names that might be technologies.
    Runs on original (not over-cleaned) text for better entity recognition.
    """
    found = set()
    doc = NLP(text[:100_000])   # spaCy limit guard — 100K chars is plenty
    for ent in doc.ents:
        if ent.label_ in NER_SKILL_LABELS:
            name = ent.text.lower().strip()
            # Skip if too short, too long, or in blocklist
            if 2 < len(name) < 40 and name not in NER_BLOCKLIST:
                if not any(b in name for b in NER_BLOCKLIST):
                    found.add(name)
    return found


def extract_skills(raw_text: str, use_ner: bool = True) -> set[str]:
    """
    Full skill extraction pipeline.
    """
    if not raw_text or not isinstance(raw_text, str):
        return set()

    cleaned = clean_text(raw_text)

    skills = extract_from_dictionary(cleaned)

    if use_ner:
        ner_skills = extract_from_ner(raw_text)
        # Only add NER skills that look like real tools (not generic words)
        for s in ner_skills:
            if len(s.split()) <= 3 and s not in {"the", "a", "an"}:
                skills.add(s)

    return skills


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_resume = """
    HR Manager with 8 years of experience in talent acquisition, onboarding,
    and performance management. Proficient in Workday, SAP HR, and Excel.
    Strong background in labor law compliance and organizational development.
    """
    print("Skills found:", sorted(extract_skills(sample_resume)))
