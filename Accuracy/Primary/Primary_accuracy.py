#!/usr/bin/env python3
"""
Evaluate text2cypher Excel files with multiple similarity metrics.

Expected files in the working directory:
  - cypher-strategy-1.xlsx
  - Cypher_strategy-2.xlsx
  - cypher-strategy-3.xlsx
"""

import os
import math
import re
import collections
import difflib

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Text preprocessing & tokenization ----------

def normalize_text(s: str) -> str:
    """Normalize Cypher text: strip, lowercase, collapse spaces, remove trailing semicolon."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";")
    return s


def tokenize(s: str):
    """Tokenize normalized text into basic tokens (identifiers, keywords, etc.)."""
    s = normalize_text(s)
    tokens = re.split(r"[^a-z0-9_]+", s)
    return [t for t in tokens if t]


def token_set(s: str):
    return set(tokenize(s))


def token_multiset(s: str):
    return collections.Counter(tokenize(s))


# ---------- Similarity metrics ----------

def dice_similarity(a: str, b: str) -> float:
    """Token-level Dice similarity: 2|A∩B| / (|A| + |B|)."""
    A = token_set(a)
    B = token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return 2.0 * len(A & B) / (len(A) + len(B))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity: |A∩B| / |A∪B|."""
    A = token_set(a)
    B = token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity over token frequency vectors."""
    va = token_multiset(a)
    vb = token_multiset(b)
    if not va and not vb:
        return 1.0
    if not va or not vb:
        return 0.0

    all_tokens = set(va.keys()) | set(vb.keys())
    dot = sum(va[t] * vb[t] for t in all_tokens)

    na = math.sqrt(sum(c * c for c in va.values()))
    nb = math.sqrt(sum(c * c for c in vb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def similarity_score(a: str, b: str) -> float:
    """
    General string similarity score using difflib.SequenceMatcher.
    This is what we interpret as "Similarity Score Comparison".
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def _lcs_length(a: str, b: str) -> int:
    """Length of the Longest Common Subsequence (character-level)."""
    a = normalize_text(a)
    b = normalize_text(b)
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0

    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev_diag = 0
        for j in range(1, lb + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev_diag + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev_diag = temp
    return dp[-1]


def sequence_matching_score(a: str, b: str) -> float:
    """
    Sequence matching score based on Longest Common Subsequence.
    We normalize it similarly to a Dice-like coefficient:
      score = 2 * LCS_len / (len(a) + len(b))
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    L = _lcs_length(a_norm, b_norm)
    return 2.0 * L / (len(a_norm) + len(b_norm))


# Register all metrics in a dict for convenience
METRICS = {
    "dice_token": dice_similarity,                # 1. Lexical similarity (token-level Dice)
    "similarity_score": similarity_score,         # 2. Similarity Score Comparison (difflib)
    "sequence_matching": sequence_matching_score, # 3. Sequence matching (LCS-based)
    "jaccard": jaccard_similarity,               # 4. Jaccard (token-set)
    "cosine": cosine_similarity,                 # 5. Cosine (token frequency vectors)
}


# ---------- Helpers for reading Excel & extracting columns ----------

def find_ground_truth_column(df: pd.DataFrame) -> str:
    """Try to find the ground truth column by name."""
    for col in df.columns:
        name = col.strip().lower()
        if "ground" in name and "truth" in name:
            return col
    raise ValueError("Could not find a 'Ground Truth' column in: " + ", ".join(df.columns))


def find_question_column(df: pd.DataFrame) -> str:
    """Try to find the question column by name."""
    for col in df.columns:
        name = col.strip().lower()
        if "question" in name:
            return col
    return None


def evaluate_file(path: str, strategy_name: str) -> pd.DataFrame:
    """
    For one Excel file (one strategy), compute all metrics for all models and questions.
    Returns a DataFrame with columns:
      strategy, model, question_index, metric, score
    """
    print(f"Loading file for {strategy_name}: {path}")
    df = pd.read_excel(path)

    gt_col = find_ground_truth_column(df)
    q_col = find_question_column(df)

    excluded = {gt_col}
    if q_col is not None:
        excluded.add(q_col)
    model_cols = [c for c in df.columns if c not in excluded]

    rows = []
    for i, row in df.iterrows():
        gt = row[gt_col]
        for model in model_cols:
            pred = row[model]
            for metric_name, func in METRICS.items():
                score = func(gt, pred)
                rows.append({
                    "strategy": strategy_name,
                    "model": model,
                    "question_index": i,
                    "metric": metric_name,
                    "score": score,
                })

    return pd.DataFrame(rows)


# ---------- Plotting ----------

def plot_metric_by_strategy(agg_df: pd.DataFrame, output_dir: str):
    """
    For each metric, create a bar chart comparing strategies
    (averaged over models and questions).
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name in METRICS.keys():
        sub = agg_df[agg_df["metric"] == metric_name].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(5, 4))
        plt.bar(sub["strategy"], sub["score"])
        plt.ylim(0, 1)
        plt.xlabel("Strategy")
        plt.ylabel(metric_name)
        plt.title(f"Average {metric_name} per strategy")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{metric_name}_per_strategy.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def plot_metric_by_model(agg_df: pd.DataFrame, output_dir: str):
    """
    For each metric, create a bar chart comparing models
    (averaged over all strategies and questions).
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name in METRICS.keys():
        sub = agg_df[agg_df["metric"] == metric_name].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(7, 4))
        plt.bar(sub["model"], sub["score"])
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("LLM model")
        plt.ylabel(metric_name)
        plt.title(f"Average {metric_name} per model (all strategies)")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{metric_name}_per_model.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


# ---------- Main ----------

def main():
    # Map strategy name to Excel filename
    strategy_files = [
        ("strategy-1", "cypher-strategy-1.xlsx"),
        ("strategy-2", "Cypher_strategy-2.xlsx"),
        ("strategy-3", "cypher-strategy-3.xlsx"),
    ]

    all_results = []
    for strategy_name, filename in strategy_files:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Expected file '{filename}' for {strategy_name} not found in current directory."
            )
        df_metrics = evaluate_file(filename, strategy_name)
        all_results.append(df_metrics)

    results = pd.concat(all_results, ignore_index=True)

    # 1) Save raw results
    results.to_csv("metrics_raw_results.csv", index=False)
    print("Saved raw metric scores: metrics_raw_results.csv")

    # 2) Aggregate: mean score per strategy & metric (averaging over models and questions)
    agg_strategy = (
        results
        .groupby(["strategy", "metric"], as_index=False)["score"]
        .mean()
    )
    agg_strategy.to_csv("metrics_by_strategy.csv", index=False)
    print("Saved aggregated metric scores by strategy: metrics_by_strategy.csv")

    # >>> NEW: pivot to get one row per strategy with each metric as a column
    pivot = agg_strategy.pivot(index="strategy", columns="metric", values="score")
    # overall average across all metrics for that strategy
    pivot["overall_mean"] = pivot.mean(axis=1)
    pivot_strategy = pivot.reset_index()
    pivot_strategy.to_csv("metrics_by_strategy_pivot.csv", index=False)
    print("Saved strategy-level average scores (one row per strategy): metrics_by_strategy_pivot.csv")
    # <<< END NEW

    # 3) Aggregate: mean score per model & metric (averaging over strategies and questions)
    agg_model = (
        results
        .groupby(["model", "metric"], as_index=False)["score"]
        .mean()
    )
    agg_model.to_csv("metrics_by_model.csv", index=False)
    print("Saved aggregated metric scores by model: metrics_by_model.csv")

    # 4) Aggregate: mean score per (strategy, model, metric)
    agg_strategy_model = (
        results
        .groupby(["strategy", "model", "metric"], as_index=False)["score"]
        .mean()
    )
    agg_strategy_model.to_csv("metrics_by_strategy_and_model.csv", index=False)
    print("Saved aggregated metric scores by strategy and model: metrics_by_strategy_and_model.csv")

    # 5) Plot per strategy
    plot_metric_by_strategy(agg_strategy, output_dir="plots")

    # 6) Plot per model (overall, across strategies)
    plot_metric_by_model(agg_model, output_dir="plots_models")

    print("Done. Check CSV files and 'plots' / 'plots_models' folders.")


if __name__ == "__main__":
    main()
