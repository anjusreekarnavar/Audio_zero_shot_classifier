import os

import numpy as np
import pandas as pd

from src.encoder import AudioEncoder
from src.metrics import reciprocal_rank, recall_at_k, total_variation


REFERENCE_BANK = "cache/reference_bank.npz"
MANIFEST = "field_queries/manifest_dev.csv"
GROUND_TRUTH = "field_queries/abundance_dev.csv"
QUERY_ROOT = "field_queries"


# --------------------------------------------------
# Load reference bank
# --------------------------------------------------

bank = np.load(REFERENCE_BANK)

classes = list(bank.files)

prototypes = np.stack([
    bank[class_name]
    for class_name in classes
])

print("Reference prototypes:", prototypes.shape)


# --------------------------------------------------
# Load development data
# --------------------------------------------------

manifest = pd.read_csv(MANIFEST)
truth_df = pd.read_csv(GROUND_TRUTH)

# Make sure abundance columns are in exactly the same
# order as our reference-bank classes.
truth_df = truth_df.set_index("query_id")

encoder = AudioEncoder()


results = []


# --------------------------------------------------
# Process every development query
# --------------------------------------------------

for i, row in manifest.iterrows():

    query_id = row["query_id"]
    wav_path = os.path.join(
        QUERY_ROOT,
        row["wav"],
    )

    print(
        f"[{i + 1}/{len(manifest)}] "
        f"{query_id} "
        f"k={row['k']} "
        f"SNR={row['snr_db']}"
    )

    query_embedding = encoder.encode(wav_path)

    # Cosine similarity because vectors are normalized.
    similarities = prototypes @ query_embedding

    # --------------------------------------------------
    # Convert similarity scores into a simple abundance
    # vector.
    #
    # This is only a baseline. Negative similarities
    # cannot represent abundance, so clip them to zero
    # and normalize.
    # --------------------------------------------------

    abundance = np.maximum(similarities, 0.0)

    if abundance.sum() > 0:
        abundance = abundance / abundance.sum()
    else:
        abundance = np.ones(len(classes)) / len(classes)

    truth = truth_df.loc[
        query_id,
        classes
    ].to_numpy(dtype=float)

    rr = reciprocal_rank(
        similarities,
        truth,
    )

    recall = recall_at_k(
        similarities,
        truth,
        int(row["k"]),
    )

    tv = total_variation(
        abundance,
        truth,
    )

    results.append({
        "query_id": query_id,
        "k": int(row["k"]),
        "snr_db": float(row["snr_db"]),
        "rr": rr,
        "recall_at_k": recall,
        "tv": tv,
    })


# --------------------------------------------------
# Results
# --------------------------------------------------

results = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results.to_csv(
    "results/baseline_dev.csv",
    index=False,
)


print()
print("==============================")
print("OVERALL BASELINE")
print("==============================")

print("MRR:", results["rr"].mean())
print("Recall@k:", results["recall_at_k"].mean())
print("TV:", results["tv"].mean())


print()
print("==============================")
print("BY NUMBER OF SOURCES (k)")
print("==============================")

print(
    results.groupby("k")[
        ["rr", "recall_at_k", "tv"]
    ].mean()
)


print()
print("==============================")
print("BY SNR")
print("==============================")

print(
    results.groupby("snr_db")[
        ["rr", "recall_at_k", "tv"]
    ].mean()
)


print()
print("==============================")
print("BY k AND SNR")
print("==============================")

print(
    results.groupby(
        ["k", "snr_db"]
    )[
        ["rr", "recall_at_k", "tv"]
    ].mean()
)
