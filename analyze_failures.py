import numpy as np
import pandas as pd

PREDICTIONS = "results/final/predictions_test.csv"
TRUTH = "field_queries/abundance_test.csv"
MANIFEST = "field_queries/manifest_test.csv"

pred = pd.read_csv(PREDICTIONS).set_index("query_id")
truth = pd.read_csv(TRUTH).set_index("query_id")
manifest = pd.read_csv(MANIFEST).set_index("query_id")

classes = list(pred.columns)

rows = []

for query_id in pred.index:

    p = pred.loc[query_id, classes].to_numpy(dtype=float)
    t = truth.loc[query_id, classes].to_numpy(dtype=float)

    tv = 0.5 * np.abs(p - t).sum()

    pred_order = np.argsort(-p)
    true_indices = np.flatnonzero(t > 0)

    top_pred_idx = pred_order[0]
    top_pred = classes[top_pred_idx]

    top_correct = top_pred_idx in true_indices

    true_classes = [
        f"{classes[i]}:{t[i]:.3f}"
        for i in np.argsort(-t)
        if t[i] > 0
    ]

    predicted_classes = [
        f"{classes[i]}:{p[i]:.3f}"
        for i in pred_order[:5]
        if p[i] > 0
    ]

    rows.append({
        "query_id": query_id,
        "k": int(manifest.loc[query_id, "k"]),
        "snr_db": manifest.loc[query_id, "snr_db"],
        "tv": tv,
        "top_correct": top_correct,
        "true_classes": ", ".join(true_classes),
        "predicted_classes": ", ".join(predicted_classes),
    })

results = pd.DataFrame(rows)

print("\n==========================================")
print("HIGHEST TV ERRORS")
print("==========================================")

print(
    results.sort_values("tv", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n==========================================")
print("WRONG TOP-1 PREDICTIONS")
print("==========================================")

print(
    results[results["top_correct"] == False]
    .sort_values("tv", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n==========================================")
print("HIGH-TV MULTI-SOURCE FAILURES (k >= 2)")
print("==========================================")

print(
    results[results["k"] >= 2]
    .sort_values("tv", ascending=False)
    .head(10)
    .to_string(index=False)
)

results.to_csv("results/final/failure_analysis_test.csv", index=False)

print("\nSaved: results/final/failure_analysis_test.csv")