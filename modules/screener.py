"""
Module 3: Vectorizer & Scorer
-------------------------------
Purpose : Convert text to numbers, then compute how similar
          each resume is to the job description.

Core concept — TF-IDF + Cosine Similarity:

  TF-IDF (Term Frequency - Inverse Document Frequency)
  -------------------------------------------------------
  Every word gets a weight based on two things:
    TF  : How often it appears in THIS document (resume)
          "python" appearing 5 times > appearing 1 time
    IDF : How rare it is across ALL documents
          "python" appearing in only 20/2100 resumes is more
          informative than "experience" appearing in all 2100

  Result: a vector (array of numbers) for each document.
  "python" gets a high number in a Python developer's resume,
  near-zero in an HR manager's resume.

  Cosine Similarity
  -----------------
  Two TF-IDF vectors point in "directions" in high-dimensional space.
  Cosine similarity = cos(angle between them).
    1.0 = identical direction = perfect match
    0.0 = perpendicular = completely unrelated
    -1.0 = opposite (rare for text)

  We compute cosine similarity between the JD vector
  and each resume vector → that's the base relevance score.

  Composite Score
  ---------------
  Final score = (0.6 × cosine_sim) + (0.4 × skill_match_ratio)

  Weighting rationale:
    cosine_sim      captures overall language/context match
    skill_match     directly rewards having the required skills
    60/40 split     prevents gaming by skill-keyword stuffing
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_cleaner import clean_text
from modules.skill_extractor import extract_skills

# Weights for composite score
W_COSINE = 0.6
W_SKILL  = 0.4


class ResumeScreener:
    """
    End-to-end resume screening system.

    Workflow:
      1. fit(df)          — clean + vectorize all resumes in the dataset
      2. screen(jd_text)  — score and rank resumes against a job description
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,   # top 10K vocabulary terms
            ngram_range=(1, 2),    # unigrams + bigrams ("machine learning" as one token)
            min_df=2,              # ignore terms appearing in < 2 docs
            sublinear_tf=True,     # use log(TF) to dampen very high counts
        )
        self.resume_matrix = None   # shape: (n_resumes, n_features)
        self.df = None              # original dataframe with metadata
        self.resume_skills = []     # list of skill sets, one per resume

    # ── Step 1: Fit on resume corpus ─────────────────────────────────────────

    def fit(self, df: pd.DataFrame, text_col: str = "Resume_str") -> "ResumeScreener":
        """
        Clean all resumes and build the TF-IDF matrix.

        Args:
            df       : DataFrame with at least text_col and Category columns
            text_col : Column containing raw resume text

        Returns:
            self (for chaining)
        """
        print(f"[fit] Cleaning {len(df)} resumes...")
        self.df = df.copy().reset_index(drop=True)

        # Clean text for vectorization
        self.df["clean_text"] = self.df[text_col].apply(clean_text)

        # Extract skills (used in composite score)
        print("[fit] Extracting skills from resumes (this takes ~1-2 min)...")
        self.resume_skills = [
            extract_skills(row, use_ner=False)   # NER off for speed on bulk
            for row in self.df[text_col]
        ]

        # Build TF-IDF matrix
        print("[fit] Building TF-IDF matrix...")
        self.resume_matrix = self.vectorizer.fit_transform(self.df["clean_text"])
        print(f"[fit] Done. Matrix shape: {self.resume_matrix.shape}")
        return self

    # ── Step 2: Screen against a job description ─────────────────────────────

    def screen(
        self,
        jd_text: str,
        top_n: int = 10,
        filter_category: str = None,
    ) -> pd.DataFrame:
        """
        Score and rank all resumes against the given job description.

        Args:
            jd_text          : Full job description text
            top_n            : How many top candidates to return
            filter_category  : Optional — limit to one category label
                               e.g. "HR", "Data Science", "Java Developer"

        Returns:
            DataFrame with columns:
              rank, candidate_id, category, cosine_score,
              skill_score, final_score, matched_skills, missing_skills
        """
        if self.resume_matrix is None:
            raise RuntimeError("Call fit() before screen()")

        # --- Parse job description ---
        jd_clean   = clean_text(jd_text)
        jd_skills  = extract_skills(jd_text, use_ner=True)

        # Vectorize JD using the SAME vocabulary (transform, not fit_transform)
        jd_vector  = self.vectorizer.transform([jd_clean])

        # --- Cosine similarity (all resumes at once — fast matrix op) ---
        cosine_scores = cosine_similarity(jd_vector, self.resume_matrix)[0]

        # --- Skill match ratio ---
        skill_scores = np.array([
            self._skill_match_ratio(resume_skills, jd_skills)
            for resume_skills in self.resume_skills
        ])

        # --- Composite score ---
        final_scores = W_COSINE * cosine_scores + W_SKILL * skill_scores

        # --- Build results dataframe ---
        results = self.df[["ID", "Category"]].copy()
        results["cosine_score"] = cosine_scores
        results["skill_score"]  = skill_scores
        results["final_score"]  = final_scores
        results["matched_skills"] = [
            sorted(rs & jd_skills) for rs in self.resume_skills
        ]
        results["missing_skills"] = [
            sorted(jd_skills - rs) for rs in self.resume_skills
        ]

        # --- Optional category filter ---
        if filter_category:
            results = results[
                results["Category"].str.lower() == filter_category.lower()
            ]

        # --- Sort and return top N ---
        results = (
            results
            .sort_values("final_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        results.index += 1    # rank starts at 1
        results.index.name = "rank"

        # Round scores for readability
        for col in ["cosine_score", "skill_score", "final_score"]:
            results[col] = results[col].round(4)

        return results.reset_index()

    @staticmethod
    def _skill_match_ratio(resume_skills: set, jd_skills: set) -> float:
        """Fraction of required JD skills present in the resume."""
        if not jd_skills:
            return 0.0
        matched = len(resume_skills & jd_skills)
        return matched / len(jd_skills)
