import csv
import os
import numpy as np

from audiomae_encoder import AudioMAEEncoder


FIELD_ROOT = (
    "/export/home/s5284664/esper-audio-challenge/"
    "field_queries"
)

MANIFEST = os.path.join(
    FIELD_ROOT,
    "manifest_dev.csv",
)

DEV_DIR = os.path.join(
    FIELD_ROOT,
    "dev",
)

OUTPUT_DIR = "cache_audiomae"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dev_embeddings.npz",
)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read manifest
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("DEV queries:", len(rows))
    print("Manifest columns:", rows[0].keys())

    encoder = AudioMAEEncoder(
        checkpoint="ckpt/pretrained.pth"
    )

    embeddings = []
    query_ids = []

    for i, row in enumerate(rows, start=1):

        # Query ID is the filename stem used by the generator.
        # Example: dev_00000
        query_id = f"dev_{i - 1:05d}"

        wav_path = os.path.join(
            DEV_DIR,
            query_id + ".wav",
        )

        if not os.path.exists(wav_path):
            raise FileNotFoundError(wav_path)

        print(
            f"[{i}/{len(rows)}] {query_id}",
            flush=True,
        )

        embedding = encoder.encode(wav_path)

        embeddings.append(embedding)
        query_ids.append(query_id)

    embeddings = np.stack(embeddings)

    np.savez(
        OUTPUT_FILE,
        embeddings=embeddings,
        query_ids=np.asarray(query_ids),
    )

    print()
    print("Embeddings shape:", embeddings.shape)
    print("First query:", query_ids[0])
    print("Last query:", query_ids[-1])

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        "Embedding norm min/max:",
        norms.min(),
        norms.max(),
    )

    print("Saved:", OUTPUT_FILE)

    assert len(rows) == 960
    assert embeddings.shape == (960, 768)
    assert np.allclose(norms, 1.0, atol=1e-5)

    print("SUCCESS")


if __name__ == "__main__":
    main()
