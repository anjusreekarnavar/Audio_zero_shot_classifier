import argparse
import os

import numpy as np
import pandas as pd

from src.encoder import AudioEncoder


TOP_N = 3
RELATIVE_THRESHOLD = 0.8


def build_reference_embeddings(encoder, library_manifest, esc50_root):
    """
    Encode every clean reference clip.

    Returns
    -------
    classes : list[str]
        Class names in output order.

    reference_embeddings : dict[str, np.ndarray]
        Each class maps to an array of shape (num_references, 512).
    """

    manifest = pd.read_csv(library_manifest)

    classes = list(dict.fromkeys(manifest["category"]))
    reference_embeddings = {}

    print("\nEncoding reference library...")

    for class_name in classes:

        rows = manifest[manifest["category"] == class_name]

        embeddings = []

        for _, row in rows.iterrows():

            wav_path = os.path.join(
                esc50_root,
                row["path"],
            )

            embedding = encoder.encode(wav_path)
            embeddings.append(embedding)

        reference_embeddings[class_name] = np.stack(
            embeddings
        )

        print(
            f"{class_name}: "
            f"{reference_embeddings[class_name].shape}"
        )

    return classes, reference_embeddings


def calculate_class_scores(
    query_embedding,
    classes,
    reference_embeddings,
):
    """
    Top-3 reference aggregation.
    """

    scores = []

    for class_name in classes:

        references = reference_embeddings[class_name]

        # cosine similarity because embeddings are L2 normalized
        similarities = references @ query_embedding

        top_n = min(TOP_N, len(similarities))

        top_similarities = np.sort(
            similarities
        )[-top_n:]

        score = np.mean(top_similarities)

        scores.append(score)

    return np.asarray(scores)


def scores_to_abundance(scores):
    """
    Convert similarity scores into sparse abundance.

    Keep classes whose score is at least 80% of
    the strongest class score.
    """

    best_score = np.max(scores)

    keep = scores >= (
        RELATIVE_THRESHOLD * best_score
    )

    abundance = np.where(
        keep,
        np.maximum(scores, 0.0),
        0.0,
    )

    # Safety fallback
    if abundance.sum() <= 0:

        abundance = np.zeros_like(
            scores,
            dtype=float,
        )

        abundance[np.argmax(scores)] = 1.0

    else:

        abundance /= abundance.sum()

    return abundance


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--esc50-root",
        required=True,
    )

    parser.add_argument(
        "--library-manifest",
        required=True,
    )

    parser.add_argument(
        "--query-manifest",
        required=True,
    )

    parser.add_argument(
        "--query-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    encoder = AudioEncoder()

    # --------------------------------------------------
    # Reference library
    # --------------------------------------------------

    classes, reference_embeddings = (
        build_reference_embeddings(
            encoder,
            args.library_manifest,
            args.esc50_root,
        )
    )

    print("\nClasses:", len(classes))

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------

    query_manifest = pd.read_csv(
        args.query_manifest
    )

    predictions = []

    print("\nEncoding queries...")

    for index, row in query_manifest.iterrows():

        query_id = row["query_id"]

        wav_path = os.path.join(
            args.query_dir,
            f"{query_id}.wav",
        )

        query_embedding = encoder.encode(
            wav_path
        )

        scores = calculate_class_scores(
            query_embedding,
            classes,
            reference_embeddings,
        )

        abundance = scores_to_abundance(
            scores
        )

        prediction = {
            "query_id": query_id,
        }

        for class_name, value in zip(
            classes,
            abundance,
        ):
            prediction[class_name] = value

        predictions.append(prediction)

        if (index + 1) % 50 == 0:
            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(query_manifest)}"
            )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    predictions = pd.DataFrame(
        predictions
    )

    predictions.to_csv(
        args.output,
        index=False,
    )

    print()
    print("Saved:", args.output)

    # Verify abundance requirement
    class_values = predictions[
        classes
    ].to_numpy()

    row_sums = class_values.sum(axis=1)

    print(
        "Minimum prediction:",
        class_values.min(),
    )

    print(
        "Minimum row sum:",
        row_sums.min(),
    )

    print(
        "Maximum row sum:",
        row_sums.max(),
    )


if __name__ == "__main__":
    main()