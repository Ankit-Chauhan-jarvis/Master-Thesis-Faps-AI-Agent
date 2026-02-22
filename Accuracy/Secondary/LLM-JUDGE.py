import os
import math
from typing import Any, List, Tuple

import pandas as pd
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path

URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "Ankujarvis@1094"
DATABASE = "neo4j"


def find_ground_truth_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        name = col.strip().lower()
        if "ground" in name and "truth" in name:
            return col
    raise ValueError("Could not find a 'Ground Truth' column in: " + ", ".join(df.columns))


def find_question_column(df: Pd.DataFrame) -> str:
    for col in df.columns:
        name = col.strip().lower()
        if "question" in name:
            return col
    return None


def _normalize_scalar_value_raw(value: Any) -> Any:
    if isinstance(value, Node):
        return ("node", value.element_id)
    if isinstance(value, Relationship):
        return ("rel", value.element_id)
    if isinstance(value, Path):
        nodes = [_normalize_scalar_value_raw(n) for n in value.nodes]
        rels = [_normalize_scalar_value_raw(r) for r in value.relationships]
        return ("path", tuple(nodes), tuple(rels))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_scalar_value_raw(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize_scalar_value_raw(v)) for k, v in value.items()))
    return value


def normalize_scalar_value(value: Any) -> str:
    raw = _normalize_scalar_value_raw(value)
    return repr(raw)


def normalize_record_row(record) -> Tuple[str, ...]:
    normalized_values = [normalize_scalar_value(v) for v in record.values()]
    return tuple(sorted(normalized_values))


def normalize_result(result_list) -> List[Tuple[str, ...]]:
    rows = [normalize_record_row(rec) for rec in result_list]
    rows.sort()
    return rows


def results_equal(gt_records, pred_records) -> bool:
    norm_gt = normalize_result(gt_records)
    norm_pred = normalize_result(pred_records)
    return norm_gt == norm_pred


def run_cypher(session, query: str):
    if query is None:
        return None
    if isinstance(query, float) and math.isnan(query):
        return None

    query = str(query).strip()
    if not query:
        return None

    if query.endswith(";"):
        query = query[:-1]

    try:
        result = session.run(query)
        records = list(result)
        return records
    except Exception as e:
        print(f"[WARN] Cypher execution failed: {e}")
        print(f"       Query was: {query[:200]}{'...' if len(query) > 200 else ''}")
        return None


def evaluate_strategy_file(driver, strategy_name: str, filename: str):
    print(f"\n=== Evaluating {strategy_name} from file: {filename} ===")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_excel(filename)
    gt_col = find_ground_truth_column(df)
    q_col = find_question_column(df)

    excluded = {gt_col}
    if q_col is not None:
        excluded.add(q_col)
    model_cols = [c for c in df.columns if c not in excluded]

    print(f"Detected ground truth column: {gt_col}")
    print(f"Detected question column: {q_col}")
    print("Detected model columns:")
    for m in model_cols:
        print(f"  - {m}")

    results = {m: {"total": 0, "correct": 0} for m in model_cols}

    with driver.session(database=DATABASE) as session:
        for idx, row in df.iterrows():
            gt_query = row[gt_col]

            gt_records = run_cypher(session, gt_query)
            if gt_records is None:
                print(f"[WARN] Ground truth query failed or empty at row {idx}, skipping this row for all models.")
                continue

            for model in model_cols:
                pred_query = row[model]
                results[model]["total"] += 1

                pred_records = run_cypher(session, pred_query)
                if pred_records is None:
                    continue

                if results_equal(gt_records, pred_records):
                    results[model]["correct"] += 1

    summary_rows = []
    for model, stats in results.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0.0
        print(f"[{strategy_name}] Model: {model}")
        print(f"  Total evaluated queries: {total}")
        print(f"  Correct (same result set as GT): {correct}")
        print(f"  Execution-based semantic accuracy: {acc:.3f}")
        summary_rows.append({
            "strategy": strategy_name,
            "model": model,
            "total": total,
            "correct": correct,
            "execution_accuracy": acc,
        })

    return summary_rows


def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    strategy_files = [
        ("strategy-1", "cypher-strategy-1.xlsx"),
        ("strategy-2", "Cypher_strategy-2.xlsx"),
        ("strategy-3", "cypher-strategy-3.xlsx"),
    ]

    all_rows = []
    try:
        for strategy_name, filename in strategy_files:
            summary = evaluate_strategy_file(driver, strategy_name, filename)
            all_rows.extend(summary)
    finally:
        driver.close()

    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv("execution_accuracy_results.csv", index=False)
    print("\nSaved execution-based semantic accuracy to execution_accuracy_results.csv")


if __name__ == "__main__":
    main()
