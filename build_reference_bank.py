import os

import numpy as np

from src.encoder import AudioEncoder
from src.reference_bank import build_reference_bank


ESC50_ROOT = "/scratch/s5284664/ESC-50-master"
MANIFEST = "field_queries/manifest_library.csv"

PROTOTYPE_OUTPUT = "cache/reference_bank.npz"
EMBEDDING_OUTPUT = "cache/reference_embeddings.npz"


os.makedirs("cache", exist_ok=True)

encoder = AudioEncoder()


# Build both versions of the reference bank
prototypes, class_embeddings = build_reference_bank(
    encoder=encoder,
    manifest_path=MANIFEST,
    esc50_root=ESC50_ROOT,
)


print()
print("Reference bank built!")
print("Number of classes:", len(prototypes))


# Show their shapes
for category in prototypes:

    print(
        category,
        "prototype:",
        prototypes[category].shape,
        "individual:",
        class_embeddings[category].shape,
    )


# Save averaged prototypes
np.savez(
    PROTOTYPE_OUTPUT,
    **prototypes,
)


# Save all individual reference embeddings
np.savez(
    EMBEDDING_OUTPUT,
    **class_embeddings,
)


print()
print("Saved:")
print(PROTOTYPE_OUTPUT)
print(EMBEDDING_OUTPUT)
