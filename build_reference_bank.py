import os
import numpy as np

from src.encoder import AudioEncoder
from src.reference_bank import build_reference_bank


ESC50_ROOT = "/scratch/s5284664/ESC-50-master"
MANIFEST = "field_queries/manifest_library.csv"
OUTPUT = "cache/reference_bank.npz"


os.makedirs("cache", exist_ok=True)

encoder = AudioEncoder()

prototypes = build_reference_bank(
    encoder=encoder,
    manifest_path=MANIFEST,
    esc50_root=ESC50_ROOT,
)

print()
print("Reference bank built!")
print("Number of classes:", len(prototypes))

for category, embedding in prototypes.items():
    print(
        category,
        embedding.shape,
        "norm =",
        np.linalg.norm(embedding),
    )


# Save all 20 class prototypes.
np.savez(
    OUTPUT,
    **prototypes,
)

print()
print("Saved reference bank to:", OUTPUT)
