"""
Module 4: Report Generator
----------------------------
Purpose : Take the raw scored DataFrame and turn it into a clean,
          readable report that a recruiter or HR manager can act on.

Output formats:
  1. Console print   — formatted table with colour-coded score bands
  2. CSV export      — machine-readable for further filtering
  3. Text report     — per-candidate breakdown with skill gap analysis

Skill gap analysis per candidate:
  matched_skills  — skills the candidate HAS that the JD requires
  missing_skills  — skills the JD requires but the candidate LACKS
  extra_skills    — skills the candidate has that the JD didn't mention
                    (signals versatility / possible bonus qualifications)
"""

import os
import textwrap
from datetime import datetime
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def score_band(score: float) -> str:
    """Label a score with a human-readable quality band."""
    if score >= 0.70:
        return "Excellent"
    elif score >= 0.50:
        return "Good"
    elif score >= 0.30:
        return "Fair"
    else:
        return "Weak"


def format_skills(skills: list, max_show: int = 8) -> str:
    """Format a list of skills as a comma-separated string, truncated if long."""
    if not skills:
        return "none"
    shown = skills[:max_show]
    rest  = len(skills) - max_show
    s = ", ".join(shown)
    if rest > 0:
        s += f"  (+{rest} more)"
    return s


def print_ranked_table(results: pd.DataFrame, jd_title: str = ""):
    """Print a formatted summary table to the console."""
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  Resume Screening Results  {'| ' + jd_title if jd_title else ''}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)
    print(f"  {'Rank':<5} {'ID':<12} {'Category':<22} {'Score':<8} {'Band':<12} {'Skills matched'}")
    print(sep)
    for _, row in results.iterrows():
        band   = score_band(row["final_score"])
        skills = format_skills(row["matched_skills"], max_show=4)
        print(
            f"  {row['rank']:<5} {str(row['ID']):<12} "
            f"{str(row['Category']):<22} {row['final_score']:<8.4f} "
            f"{band:<12} {skills}"
        )
    print(sep + "\n")


def generate_candidate_reports(
    results: pd.DataFrame,
    jd_text: str,
    jd_skills: set,
    jd_title: str = "Job",
) -> str:
    """
    Generate a detailed per-candidate breakdown as a multi-line string.

    Returns the full report text (also saved to file).
    """
    lines = []
    header = f"RESUME SCREENING REPORT — {jd_title.upper()}"
    lines.append("=" * 70)
    lines.append(header)
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Required skills identified in JD: {len(jd_skills)}")
    lines.append(f"  -> {format_skills(sorted(jd_skills), max_show=12)}")
    lines.append("=" * 70)

    for _, row in results.iterrows():
        band     = score_band(row["final_score"])
        matched  = row["matched_skills"]
        missing  = row["missing_skills"]
        coverage = (
            f"{len(matched)}/{len(jd_skills)}" if jd_skills else "N/A"
        )

        lines.append(f"\nRank #{row['rank']}  |  ID: {row['ID']}  |  Category: {row['Category']}")
        lines.append(f"  Final score    : {row['final_score']:.4f}  [{band}]")
        lines.append(f"  Cosine match   : {row['cosine_score']:.4f}   (language/context similarity)")
        lines.append(f"  Skill match    : {row['skill_score']:.4f}   (required skills coverage)")
        lines.append(f"  Skill coverage : {coverage} required skills present")
        lines.append(f"  Matched skills : {format_skills(matched)}")
        lines.append(f"  Missing skills : {format_skills(missing)}")

        # Actionable recommendation
        if row["final_score"] >= 0.70:
            rec = "SHORTLIST — Strong match. Recommend immediate interview."
        elif row["final_score"] >= 0.50:
            rec = "CONSIDER  — Good potential. Review experience depth before deciding."
        elif row["final_score"] >= 0.30:
            rec = "REVIEW    — Partial match. May suit junior role or adjacent position."
        else:
            rec = "PASS      — Weak alignment with this specific role."
        lines.append(f"  Recommendation : {rec}")
        lines.append("  " + "─" * 66)

    report_text = "\n".join(lines)
    return report_text


def save_report(report_text: str, results: pd.DataFrame, jd_title: str = "job"):
    """Save report as .txt and results as .csv to the output folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = jd_title.lower().replace(" ", "_")[:30]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = os.path.join(OUTPUT_DIR, f"report_{slug}_{ts}.txt")
    csv_path = os.path.join(OUTPUT_DIR, f"results_{slug}_{ts}.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Flatten list columns for CSV
    csv_df = results.copy()
    csv_df["matched_skills"] = csv_df["matched_skills"].apply(lambda x: "; ".join(x))
    csv_df["missing_skills"] = csv_df["missing_skills"].apply(lambda x: "; ".join(x))
    csv_df.to_csv(csv_path, index=False)

    print(f"[report] Saved text report : {txt_path}")
    print(f"[report] Saved CSV results : {csv_path}")
    return txt_path, csv_path
