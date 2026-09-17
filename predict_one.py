import numpy as np

from src.encoder import AudioEncoder


REFERENCE_BANK = "cache/reference_bank.npz"
QUERY = "field_queries/dev/dev_00000.wav"


# --------------------------------------------------
# Load reference prototypes
# --------------------------------------------------

bank = np.load(REFERENCE_BANK)

classes = list(bank.files)

prototypes = np.stack([
    bank[class_name]
    for class_name in classes
])


print("Loaded reference bank.")
print("Classes:", len(classes))
print("Prototype matrix:", prototypes.shape)


# --------------------------------------------------
# Encode query
# --------------------------------------------------

encoder = AudioEncoder()

query_embedding = encoder.encode(QUERY)

print()
print("Query embedding:", query_embedding.shape)


# --------------------------------------------------
# Cosine similarities
#
# Everything is L2-normalized, therefore the dot
# product is cosine similarity.
# --------------------------------------------------

similarities = prototypes @ query_embedding


# Highest similarity first.
ranking = np.argsort(-similarities)


print()
print("Ranking for dev_00000:")
print("----------------------")

for rank, index in enumerate(ranking, start=1):
    print(
        f"{rank:2d}. "
        f"{classes[index]:20s} "
        f"{similarities[index]:.4f}"
    )
