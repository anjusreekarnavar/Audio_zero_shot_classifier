import numpy as np
import pandas as pd

from src.metrics import reciprocal_rank, recall_at_k, total_variation


DEV_EMBEDDINGS = "cache/dev_embeddings.npz"
REFERENCE_EMBEDDINGS = "cache/reference_embeddings.npz"
MANIFEST = "field_queries/manifest_dev.csv"
GROUND_TRUTH = "field_queries/abundance_dev.csv"
OUTPUT = "results/max8_dev.csv"


# --------------------------------------------------
# Load cached query embeddings
# --------------------------------------------------

dev_data = np.load(DEV_EMBEDDINGS)

query_ids = dev_data["query_ids"]
query_embeddings = dev_data["embeddings"]

print("Queries:", query_embeddings.shape)


# --------------------------------------------------
# Load the 8 references for every class
# --------------------------------------------------

reference_bank = np.load(REFERENCE_EMBEDDINGS)

classes = list(reference_bank.files)

print("Classes:", len(classes))


# --------------------------------------------------
# Load metadata and ground truth
# --------------------------------------------------

manifest = pd.read_csv(MANIFEST).set_index("query_id")
truth_df = pd.read_csv(GROUND_TRUTH).set_index("query_id")


results = []


# --------------------------------------------------
# Evaluate all 960 queries
# --------------------------------------------------

for i, query_id in enumerate(query_ids):

    query_id = str(query_id)
    query = query_embeddings[i]

    class_scores = []

    # Calculate Max-of-8 score for every class
    for class_name in classes:

        # Shape = (8, 512)
        references = reference_bank[class_name]

        # 8 cosine similarities
        similarities = references @ query

        # Best matching reference
        score = np.max(similarities)

        class_scores.append(score)

    class_scores = np.array(class_scores)


    # --------------------------------------------------
    # Simple abundance baseline
    # --------------------------------------------------

    abundance = np.maximum(class_scores, 0.0)

    if abundance.sum() > 0:
        abundance /= abundance.sum()
    else:
        abundance = np.ones(len(classes)) / len(classes)


    # Ground-truth abundance
    truth = truth_df.loc[
        query_id,
        classes
    ].to_numpy(dtype=float)


    k = int(
        manifest.loc[query_id, "k"]
    )

    snr_db = float(
        manifest.loc[query_id, "snr_db"]
    )


    # Metrics
    rr = reciprocal_rank(
        class_scores,
        truth,
    )

    recall = recall_at_k(
        class_scores,
        truth,
        k,
    )

    tv = total_variation(
        abundance,
        truth,
    )


    results.append({
        "query_id": query_id,
        "k": k,
        "snr_db": snr_db,
        "rr": rr,
        "recall_at_k": recall,
        "tv": tv,
    })


# --------------------------------------------------
# Convert results to DataFrame
# --------------------------------------------------

results = pd.DataFrame(results)

results.to_csv(
    OUTPUT,
    index=False,
)


# --------------------------------------------------
# Report results
# --------------------------------------------------

print()
print("==============================")
print("MAX-OF-8: OVERALL")
print("==============================")

print("MRR:", results["rr"].mean())
print("Recall@k:", results["recall_at_k"].mean())
print("TV:", results["tv"].mean())


print()
print("==============================")
print("MAX-OF-8: BY k")
print("==============================")

print(
    results.groupby("k")[
        ["rr", "recall_at_k", "tv"]
    ].mean()
)


print()
print("==============================")
print("MAX-OF-8: BY SNR")
print("==============================")

print(
    results.groupby("snr_db")[
        ["rr", "recall_at_k", "tv"]
    ].mean()
)
