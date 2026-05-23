"""
main.py — Resume Screening System
===================================
Entry point. Ties all 4 modules together into a single runnable pipeline.

Usage:
  python main.py

The script:
  1. Loads Resume.csv
  2. Fits the screener on all 2100 resumes
  3. Runs three example job descriptions
  4. Prints ranked results + saves reports to /output

To screen YOUR OWN job description:
  Replace the JOB_DESCRIPTIONS dict entries with your own JD text,
  or call screener.screen(your_jd_text, top_n=10) directly.
"""

import os
import sys
import pandas as pd

# Add project root to path so module imports work
sys.path.insert(0, os.path.dirname(__file__))

from modules.screener  import ResumeScreener
from modules.reporter  import print_ranked_table, generate_candidate_reports, save_report
from modules.skill_extractor import extract_skills

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Resume.csv")
TOP_N     = 10    # candidates to return per job


# ── Sample Job Descriptions ──────────────────────────────────────────────────
# These represent three different job categories present in Resume.csv

JOB_DESCRIPTIONS = {

    "HR Manager": """
    We are looking for an experienced HR Manager to lead our Human Resources
    department. The ideal candidate will have strong experience in talent
    acquisition, recruitment, onboarding, employee relations, and performance
    management. Proficiency in HRIS tools such as Workday or SAP HR is required.
    The candidate must be familiar with labor law, compensation planning,
    benefits administration, and organizational development. Excellent
    communication and leadership skills are essential. A minimum of 5 years
    experience in HR roles is required. Experience with HR analytics is a plus.
    """,

    "Data Scientist": """
    We are hiring a Data Scientist with strong proficiency in Python, machine
    learning, and natural language processing. The role requires hands-on
    experience with scikit-learn, pandas, numpy, and deep learning frameworks
    such as TensorFlow or PyTorch. You will work on data analysis, feature
    engineering, model deployment, and data visualization. SQL and database
    skills are essential. Experience with cloud platforms (AWS or GCP) and
    version control via git is expected. Strong communication skills for
    presenting findings to non-technical stakeholders are important.
    """,

    "Java Developer": """
    Seeking a senior Java Developer to build and maintain backend services.
    Required skills include Java, Spring Boot, REST API design, SQL, and
    microservices architecture. Experience with Docker, Kubernetes, and CI/CD
    pipelines is strongly preferred. The candidate should be familiar with
    agile/scrum methodologies, git, and code review practices. Strong
    problem solving skills and the ability to work in a collaborative team
    environment are required. Experience with AWS or Azure is a bonus.
    """,
}


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load Resume.csv with error handling."""
    if not os.path.exists(path):
        print(f"[ERROR] Resume.csv not found at: {path}")
        print("        Place Resume.csv inside the /data folder and re-run.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[data]  Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
    print(f"[data]  Categories: {sorted(df['Category'].unique())[:8]}...")
    return df


def run_screening(screener: ResumeScreener, jd_title: str, jd_text: str):
    """Run one full screening cycle and output results."""
    print(f"\n{'='*60}")
    print(f"  Screening for: {jd_title}")
    print(f"{'='*60}")

    # Get required skills from JD (for gap analysis)
    jd_skills = extract_skills(jd_text, use_ner=True)
    print(f"  JD skills identified: {sorted(jd_skills)}\n")

    # Screen resumes
    results = screener.screen(jd_text, top_n=TOP_N)

    # Print summary table
    print_ranked_table(results, jd_title=jd_title)

    # Generate and save full report
    report = generate_candidate_reports(results, jd_text, jd_skills, jd_title)
    print(report)
    save_report(report, results, jd_title)


def main():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Fit screener on all resumes (done ONCE — reuse for all JDs)
    screener = ResumeScreener()
    screener.fit(df, text_col="Resume_str")

    # 3. Screen for each job description
    for jd_title, jd_text in JOB_DESCRIPTIONS.items():
        run_screening(screener, jd_title, jd_text)

    print("\n[done] All reports saved to /output folder.")


if __name__ == "__main__":
    main()
