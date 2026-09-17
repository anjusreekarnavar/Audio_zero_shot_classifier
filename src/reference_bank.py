import os

import numpy as np
import pandas as pd


def build_reference_bank(
    encoder,
    manifest_path,
    esc50_root,
):
    """
    Encode all reference WAV files.

    Returns:
        prototypes:
            One averaged embedding per class.

        class_embeddings:
            All 8 individual embeddings per class.
    """

    manifest = pd.read_csv(manifest_path)

    class_embeddings = {}

    # Encode all 160 reference WAV files
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

        # Create a list for this class if it doesn't exist
        if category not in class_embeddings:
            class_embeddings[category] = []

        # KEEP the individual embedding
        class_embeddings[category].append(embedding)

    # Convert each list into a NumPy matrix.
    #
    # For example:
    #
    # car_horn -> (8, 512)
    # dog      -> (8, 512)
    #
    for category in class_embeddings:

        class_embeddings[category] = np.stack(
            class_embeddings[category]
        )

    # Also create our original mean prototypes.
    #
    # car_horn (8, 512)
    #       ↓ average
    # car_horn (512,)
    #
    prototypes = {}

    for category, embeddings in class_embeddings.items():

        prototype = embeddings.mean(axis=0)

        # Normalize the averaged vector
        prototype = prototype / np.linalg.norm(prototype)

        prototypes[category] = prototype

    return prototypes, class_embeddings
