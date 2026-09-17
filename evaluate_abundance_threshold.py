import os
import numpy as np
import pandas as pd

from src.metrics import total_variation


DEV_EMBEDDINGS = "cache/dev_embeddings.npz"
REFERENCE_EMBEDDINGS = "cache/reference_embeddings.npz"

MANIFEST = "field_queries/manifest_dev.csv"
GROUND_TRUTH = "field_queries/abundance_dev.csv"

OUTPUT_DIR = "results"

# Top-3 reference aggregation
TOP_N = 3

# Keep a class if:
# score >= threshold * best_score
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

dev_data = np.load(DEV_EMBEDDINGS)

query_ids = dev_data["query_ids"]
query_embeddings = dev_data["embeddings"]

reference_bank = np.load(REFERENCE_EMBEDDINGS)
classes = list(reference_bank.files)

manifest = pd.read_csv(MANIFEST).set_index("query_id")
truth_df = pd.read_csv(GROUND_TRUTH).set_index("query_id")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Queries:", query_embeddings.shape)
print("Classes:", len(classes))


# ---------------------------------------------------------
# Calculate Top-3 class scores once
# ---------------------------------------------------------

all_scores = []

for query in query_embeddings:

    class_scores = []

    for class_name in classes:

        references = reference_bank[class_name]

        # similarity against all 8 references
        similarities = references @ query

        # top 3 reference similarities
        top_similarities = np.sort(similarities)[-TOP_N:]

        score = np.mean(top_similarities)

        class_scores.append(score)

    all_scores.append(class_scores)

all_scores = np.asarray(all_scores)

print("Score matrix:", all_scores.shape)


# ---------------------------------------------------------
# Try different relative thresholds
# ---------------------------------------------------------
summary = []
snr_summary = []

for threshold in THRESHOLDS:

    tv_values = []
    tv_multi_values = []
    active_counts = []

    # Store TV values separately for each SNR
    tv_by_snr = {
        -5.0: [],
        0.0: [],
        10.0: [],
    }
    tv_by_k = {
    1: [],
    2: [],
    3: [],
}

    for query_id, scores in zip(query_ids, all_scores):

        query_id = str(query_id)

        best_score = np.max(scores)

        # Relative threshold
        keep = scores >= threshold * best_score

        abundance = np.where(
            keep,
            np.maximum(scores, 0.0),
            0.0,
        )

        # Safety fallback
        if abundance.sum() <= 0:
            abundance = np.zeros(len(classes))
            abundance[np.argmax(scores)] = 1.0
        else:
            abundance /= abundance.sum()

        truth = truth_df.loc[
            query_id,
            classes
        ].to_numpy(dtype=float)

        tv = total_variation(
            abundance,
            truth,
        )

        k = int(
            manifest.loc[query_id, "k"]
        )
        tv_by_k[k].append(tv)

        snr = float(
            manifest.loc[query_id, "snr_db"]
        )

        tv_values.append(tv)

        active_counts.append(
            np.count_nonzero(abundance)
        )

        if k >= 2:
            tv_multi_values.append(tv)

        tv_by_snr[snr].append(tv)

    summary.append({
    "threshold": threshold,
    "tv_overall": np.mean(tv_values),
    "tv_k1": np.mean(tv_by_k[1]),
    "tv_k2": np.mean(tv_by_k[2]),
    "tv_k3": np.mean(tv_by_k[3]),
    "tv_k_ge_2": np.mean(tv_multi_values),
    "avg_predicted_classes": np.mean(active_counts),
})

    for snr in [-5.0, 0.0, 10.0]:

        snr_summary.append({
            "threshold": threshold,
            "snr_db": snr,
            "tv": np.mean(tv_by_snr[snr]),
        })
summary = pd.DataFrame(summary)
snr_summary = pd.DataFrame(snr_summary)

print()
print("==============================")
print("ABUNDANCE THRESHOLD RESULTS")
print("==============================")
print(summary.to_string(index=False))

print()
print("==============================")
print("TV BY SNR")
print("==============================")
print(snr_summary.to_string(index=False))

summary.to_csv(
    "results/abundance_threshold_dev.csv",
    index=False,
)

snr_summary.to_csv(
    "results/abundance_threshold_snr_dev.csv",
    index=False,
)
