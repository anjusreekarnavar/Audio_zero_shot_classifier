import numpy as np
import pandas as pd

QUERY_IDS = [
    "test_00534",
    "test_00885",
    "test_00646",
]

TOP_N = 3
THRESHOLD = 0.8

# Cached individual clean-reference embeddings:
# each class has shape (8, 512)
ref_data = np.load("cache/reference_embeddings.npz")

# We did not cache test embeddings, so this script will encode only
# the three selected queries.
from src.encoder import AudioEncoder

encoder = AudioEncoder()

truth = pd.read_csv(
    "field_queries/abundance_test.csv"
).set_index("query_id")

classes = list(ref_data.files)

for query_id in QUERY_IDS:

    print("\n" + "=" * 70)
    print(query_id)
    print("=" * 70)

    wav_path = f"field_queries/test/{query_id}.wav"

    q = encoder.encode(wav_path)

    scores = {}

    for category in classes:

        refs = ref_data[category]       # (8, 512)

        similarities = refs @ q

        top = np.sort(similarities)[-TOP_N:]

        scores[category] = float(top.mean())

    # Sort classes by raw Top-3 score
    ranking = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_score = ranking[0][1]
    cutoff = THRESHOLD * best_score

    print(f"\nBest raw score: {best_score:.4f}")
    print(f"0.8 threshold:  {cutoff:.4f}")

    print("\nRAW TOP-3 SIMILARITY RANKING")
    print("---------------------------------------------")
    print(f"{'rank':<6}{'class':<22}{'score':<12}{'kept'}")

    for rank, (category, score) in enumerate(ranking, start=1):

        kept = score >= cutoff and score > 0

        marker = "YES" if kept else "no"

        print(
            f"{rank:<6}"
            f"{category:<22}"
            f"{score:<12.4f}"
            f"{marker}"
        )

    print("\nGROUND TRUTH")
    print("---------------------------------------------")

    t = truth.loc[query_id]

    true_classes = [
        (category, float(t[category]))
        for category in classes
        if t[category] > 0
    ]

    true_classes.sort(key=lambda x: x[1], reverse=True)

    for category, abundance in true_classes:
        rank = next(
            i
            for i, (name, _) in enumerate(ranking, start=1)
            if name == category
        )

        print(
            f"{category:<22}"
            f"abundance={abundance:.3f}  "
            f"raw_rank={rank:<3} "
            f"raw_score={scores[category]:.4f}"
        )