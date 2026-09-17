import os

import numpy as np
import pandas as pd

from src.encoder import AudioEncoder


MANIFEST = "field_queries/manifest_dev.csv"
QUERY_ROOT = "field_queries"
OUTPUT = "cache/dev_embeddings.npz"


manifest = pd.read_csv(MANIFEST)

encoder = AudioEncoder()

query_ids = []
embeddings = []


for index, row in manifest.iterrows():

    query_id = row["query_id"]

    wav_path = os.path.join(
        QUERY_ROOT,
        row["wav"],
    )

    print(
        f"[{index + 1}/{len(manifest)}] "
        f"Encoding {query_id}"
    )

    embedding = encoder.encode(wav_path)

    query_ids.append(query_id)
    embeddings.append(embedding)


# Convert 960 individual (512,) vectors into
# one (960, 512) matrix.
embeddings = np.stack(embeddings)


print()
print("Query IDs:", len(query_ids))
print("Embedding matrix:", embeddings.shape)


os.makedirs("cache", exist_ok=True)

np.savez(
    OUTPUT,
    query_ids=np.array(query_ids),
    embeddings=embeddings,
)


print()
print("Saved:", OUTPUT)
