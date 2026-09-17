import os

import numpy as np
import pandas as pd


def build_reference_bank(
    encoder,
    manifest_path,
    esc50_root,
):
    """
    Encode all reference WAV files and create one prototype
    embedding per class.
    """

    manifest = pd.read_csv(manifest_path)

    class_embeddings = {}

    for index, row in manifest.iterrows():

        category = row["category"]
        relative_path = row["path"]

        wav_path = os.path.join(
            esc50_root,
            relative_path,
        )

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"Encoding {category}: {wav_path}"
        )

        embedding = encoder.encode(wav_path)

        if category not in class_embeddings:
            class_embeddings[category] = []

        class_embeddings[category].append(embedding)

    # Build one mean prototype per class.
    prototypes = {}

    for category, embeddings in class_embeddings.items():

        embeddings = np.stack(embeddings)

        prototype = embeddings.mean(axis=0)

        # Normalize the mean embedding.
        prototype = prototype / np.linalg.norm(prototype)

        prototypes[category] = prototype

    return prototypes
