import numpy as np

from src.encoder import AudioEncoder


REFERENCE_EMBEDDINGS = "cache/reference_embeddings.npz"
QUERY = "field_queries/dev/dev_00000.wav"


# --------------------------------------------------
# Load all individual reference embeddings
# --------------------------------------------------

bank = np.load(REFERENCE_EMBEDDINGS)

classes = list(bank.files)

print("Number of classes:", len(classes))

for class_name in classes:
    print(
        class_name,
        bank[class_name].shape,
    )


# --------------------------------------------------
# Encode the noisy query
# --------------------------------------------------

encoder = AudioEncoder()

query_embedding = encoder.encode(QUERY)

print()
print("Query embedding:", query_embedding.shape)


# --------------------------------------------------
# Compare query against all 8 references of each class
# --------------------------------------------------

class_scores = []

for class_name in classes:

    # Shape: (8, 512)
    reference_embeddings = bank[class_name]

    # Query is (512,)
    #
    # Result:
    #     (8, 512) @ (512,)
    #          ↓
    #         (8,)
    #
    # One similarity for each reference recording.
    similarities = reference_embeddings @ query_embedding

    # Use the best matching reference as the class score.
    score = np.max(similarities)

    class_scores.append(score)


class_scores = np.array(class_scores)


# --------------------------------------------------
# Rank classes
# --------------------------------------------------

ranking = np.argsort(-class_scores)


print()
print("MAX-OF-8 ranking for dev_00000")
print("--------------------------------")

for rank, index in enumerate(ranking, start=1):

    print(
        f"{rank:2d}. "
        f"{classes[index]:20s} "
        f"{class_scores[index]:.4f}"
    )
