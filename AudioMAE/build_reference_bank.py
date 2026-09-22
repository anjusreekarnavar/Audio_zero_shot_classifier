import csv
import os
import numpy as np

from audiomae_encoder import AudioMAEEncoder


ESC50_ROOT = "/scratch/s5284664/ESC-50-master"

MANIFEST = (
    "/export/home/s5284664/esper-audio-challenge/"
    "field_queries/manifest_library.csv"
)

OUTPUT_DIR = "cache_audiomae"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "reference_embeddings.npz",
)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    encoder = AudioMAEEncoder(
        checkpoint="ckpt/pretrained.pth"
    )

    categories = []
    paths = []
    embeddings = []

    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print()
    print("Reference clips:", len(rows))

    for i, row in enumerate(rows, start=1):

        category = row["category"]

        wav_path = os.path.join(
            ESC50_ROOT,
            row["path"],
        )

        print(
            f"[{i}/{len(rows)}] "
            f"{category}: {wav_path}"
        )

        embedding = encoder.encode(wav_path)

        categories.append(category)
        paths.append(wav_path)
        embeddings.append(embedding)

    embeddings = np.stack(embeddings)

    categories = np.asarray(categories)
    paths = np.asarray(paths)

    print()
    print("Embeddings shape:", embeddings.shape)

    np.savez(
        OUTPUT_FILE,
        embeddings=embeddings,
        categories=categories,
        paths=paths,
    )

    print("Saved:", OUTPUT_FILE)

    # Basic validation
    assert embeddings.shape[1] == 768
    assert len(embeddings) == len(categories)
    assert len(embeddings) == len(paths)

    unique_classes, counts = np.unique(
        categories,
        return_counts=True,
    )

    print()
    print("Number of classes:", len(unique_classes))
    print("References per class:")

    for category, count in zip(
        unique_classes,
        counts,
    ):
        print(f"  {category}: {count}")

    print()
    print("SUCCESS")


if __name__ == "__main__":
    main()
