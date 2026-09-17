import argparse
import numpy as np
import pandas as pd

from src.metrics import (
    reciprocal_rank,
    recall_at_k,
    total_variation,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--manifest", required=True)

    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    truth = pd.read_csv(args.truth)
    manifest = pd.read_csv(args.manifest)

    predictions = predictions.set_index("query_id")
    truth = truth.set_index("query_id")
    manifest = manifest.set_index("query_id")

    classes = list(predictions.columns)

    rows = []

    for query_id in predictions.index:

        pred = predictions.loc[
            query_id,
            classes
        ].to_numpy(dtype=float)

        target = truth.loc[
            query_id,
            classes
        ].to_numpy(dtype=float)

        k = int(
            manifest.loc[query_id, "k"]
        )

        snr = float(
            manifest.loc[query_id, "snr_db"]
        )

        rr = reciprocal_rank(
            pred,
            target,
        )

        recall = recall_at_k(
            pred,
            target,
            k,
        )

        tv = total_variation(
            pred,
            target,
        )

        top1_correct = np.argmax(pred) in np.flatnonzero(
            target > 0
        )

        rows.append({
            "query_id": query_id,
            "k": k,
            "snr_db": snr,
            "rr": rr,
            "recall_at_k": recall,
            "tv": tv,
            "top1_correct": top1_correct,
        })

    results = pd.DataFrame(rows)

    print()
    print("==============================")
    print("FINAL DEV RESULTS")
    print("==============================")

    print("MRR:", results["rr"].mean())
    print(
        "Recall@k:",
        results["recall_at_k"].mean(),
    )
    print(
        "TV:",
        results["tv"].mean(),
    )

    # Required k=1 top-1 accuracy
    k1 = results[
        results["k"] == 1
    ]

    print(
        "Top-1 accuracy (k=1):",
        k1["top1_correct"].mean(),
    )

    # Required multi-source TV
    multi = results[
        results["k"] >= 2
    ]

    print(
        "TV (k>=2):",
        multi["tv"].mean(),
    )

    print()
    print("==============================")
    print("BY k")
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


if __name__ == "__main__":
    main()