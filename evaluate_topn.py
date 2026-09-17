import os

import numpy as np
import pandas as pd

from src.metrics import reciprocal_rank, recall_at_k, total_variation


DEV_EMBEDDINGS = "cache/dev_embeddings.npz"
REFERENCE_EMBEDDINGS = "cache/reference_embeddings.npz"

MANIFEST = "field_queries/manifest_dev.csv"
GROUND_TRUTH = "field_queries/abundance_dev.csv"

OUTPUT_DIR = "results"


# ---------------------------------------------------------
# Load cached development query embeddings
# ---------------------------------------------------------

dev_data = np.load(DEV_EMBEDDINGS)

query_ids = dev_data["query_ids"]
query_embeddings = dev_data["embeddings"]

print("Queries:", query_embeddings.shape)


# ---------------------------------------------------------
# Load the 8 reference embeddings for each class
# ---------------------------------------------------------

reference_bank = np.load(REFERENCE_EMBEDDINGS)

classes = list(reference_bank.files)

print("Classes:", len(classes))


# ---------------------------------------------------------
# Load manifest and ground truth
# ---------------------------------------------------------

manifest = pd.read_csv(MANIFEST).set_index("query_id")
truth_df = pd.read_csv(GROUND_TRUTH).set_index("query_id")


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Evaluate Top-N
# ---------------------------------------------------------

for top_n in [2, 3]:

    results = []

    print()
    print("==============================")
    print(f"EVALUATING TOP-{top_n}")
    print("==============================")


    for query_id, query in zip(query_ids, query_embeddings):

        # np.str_ -> normal Python string
        query_id = str(query_id)

        class_scores = []

        for class_name in classes:

            # Shape: (8, 512)
            references = reference_bank[class_name]

            # Similarity against all 8 references
            similarities = references @ query

            # Select the N largest similarities
            top_similarities = np.sort(similarities)[-top_n:]

            # Average them
            score = np.mean(top_similarities)

            class_scores.append(score)


        class_scores = np.asarray(class_scores)


        # -------------------------------------------------
        # Convert scores into abundance estimate
        # -------------------------------------------------

        abundance = np.maximum(class_scores, 0.0)

        if abundance.sum() > 0:
            abundance /= abundance.sum()
        else:
            abundance = np.ones(len(classes)) / len(classes)


        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        truth = truth_df.loc[
            query_id,
            classes
        ].to_numpy(dtype=float)


        k = int(manifest.loc[query_id, "k"])
        snr_db = float(manifest.loc[query_id, "snr_db"])


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

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


    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results = pd.DataFrame(results)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"top{top_n}_dev.csv",
    )

    results.to_csv(
        output_file,
        index=False,
    )


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print(f"TOP-{top_n}: OVERALL")

    print("MRR:", results["rr"].mean())
    print(
        "Recall@k:",
        results["recall_at_k"].mean(),
    )
    print("TV:", results["tv"].mean())


    print()
    print(f"TOP-{top_n}: BY k")

    print(
        results.groupby("k")[
            ["rr", "recall_at_k", "tv"]
        ].mean()
    )


    print()
    print(f"TOP-{top_n}: BY SNR")

    print(
        results.groupby("snr_db")[
            ["rr", "recall_at_k", "tv"]
        ].mean()
    )
