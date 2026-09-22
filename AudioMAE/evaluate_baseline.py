import csv
import numpy as np


REFERENCE_FILE = "cache_audiomae/reference_embeddings.npz"
DEV_FILE = "cache_audiomae/dev_embeddings.npz"

ABUNDANCE_FILE = (
    "/export/home/s5284664/esper-audio-challenge/"
    "field_queries/abundance_dev.csv"
)

MANIFEST_FILE = (
    "/export/home/s5284664/esper-audio-challenge/"
    "field_queries/manifest_dev.csv"
)


def reciprocal_rank(prediction, truth):
    relevant = truth > 0
    ranking = np.argsort(-prediction)

    for rank, index in enumerate(ranking, start=1):
        if relevant[index]:
            return 1.0 / rank

    return 0.0


def recall_at_k(prediction, truth, k):
    relevant = np.flatnonzero(truth > 0)

    if len(relevant) == 0:
        return 0.0

    top_k = np.argsort(-prediction)[:k]

    hits = np.intersect1d(
        top_k,
        relevant,
    ).size

    return hits / len(relevant)


def total_variation(prediction, truth):
    return 0.5 * np.abs(
        prediction - truth
    ).sum()


# --------------------------------------------------
# Load reference embeddings
# --------------------------------------------------

reference_data = np.load(
    REFERENCE_FILE,
    allow_pickle=True,
)

reference_embeddings = reference_data["embeddings"]
reference_categories = reference_data["categories"]

print(
    "Reference embeddings:",
    reference_embeddings.shape,
)


# --------------------------------------------------
# Load DEV embeddings
# --------------------------------------------------

dev_data = np.load(
    DEV_FILE,
    allow_pickle=True,
)

dev_embeddings = dev_data["embeddings"]
query_ids = dev_data["query_ids"]

print(
    "DEV embeddings:",
    dev_embeddings.shape,
)


# --------------------------------------------------
# Load ground truth
# --------------------------------------------------

with open(ABUNDANCE_FILE, newline="") as f:
    reader = csv.DictReader(f)

    class_names = reader.fieldnames[1:]

    abundance_rows = list(reader)


truth_by_id = {}

for row in abundance_rows:
    truth_by_id[row["query_id"]] = np.array(
        [
            float(row[name])
            for name in class_names
        ],
        dtype=np.float32,
    )


print("Classes:", len(class_names))


# --------------------------------------------------
# Load manifest metadata
# --------------------------------------------------

with open(MANIFEST_FILE, newline="") as f:
    reader = csv.DictReader(f)
    manifest_rows = list(reader)


metadata = {
    row["query_id"]: {
        "k": int(row["k"]),
        "snr": float(row["snr_db"]),
    }
    for row in manifest_rows
}


# --------------------------------------------------
# Build one mean prototype per class
# --------------------------------------------------

prototypes = []

for class_name in class_names:

    mask = reference_categories == class_name

    class_references = reference_embeddings[mask]

    assert class_references.shape[0] == 8

    prototype = class_references.mean(axis=0)

    # Normalize prototype
    prototype /= np.linalg.norm(prototype)

    prototypes.append(prototype)


prototypes = np.stack(prototypes)

print(
    "Prototype shape:",
    prototypes.shape,
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

rr_values = []
recall_values = []
tv_values = []
top1_k1_values = []

results_by_k = {
    1: [],
    2: [],
    3: [],
}

results_by_snr = {
    -5.0: [],
    0.0: [],
    10.0: [],
}


for query_id, query_embedding in zip(
    query_ids,
    dev_embeddings,
):

    truth = truth_by_id[str(query_id)]

    # Embeddings and prototypes are normalized,
    # therefore dot product = cosine similarity.
    scores = prototypes @ query_embedding

    # Convert similarities to a nonnegative
    # abundance vector.
    prediction = np.maximum(scores, 0.0)

    if prediction.sum() > 0:
        prediction /= prediction.sum()
    else:
        prediction[:] = 1.0 / len(prediction)

    true_k = metadata[str(query_id)]["k"]
    snr = metadata[str(query_id)]["snr"]

    rr = reciprocal_rank(
        prediction,
        truth,
    )

    # Challenge Recall@k:
    # use the true number of sources as k.
    recall = recall_at_k(
        prediction,
        truth,
        true_k,
    )

    tv = total_variation(
        prediction,
        truth,
    )

    rr_values.append(rr)
    recall_values.append(recall)
    tv_values.append(tv)

    results_by_k[true_k].append(
        (rr, recall, tv)
    )

    results_by_snr[snr].append(
        (rr, recall, tv)
    )

    if true_k == 1:
        predicted_class = np.argmax(prediction)
        true_class = np.argmax(truth)

        top1_k1_values.append(
            float(predicted_class == true_class)
        )


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("=== AudioMAE Mean-Prototype DEV Baseline ===")

print("MRR:", np.mean(rr_values))
print("Recall:", np.mean(recall_values))
print("TV:", np.mean(tv_values))
print(
    "Top1 k=1:",
    np.mean(top1_k1_values),
)


print()
print("=== By mixture size k ===")

for k in [1, 2, 3]:

    values = np.array(
        results_by_k[k]
    )

    print(
        f"k={k}: "
        f"MRR={values[:,0].mean():.6f} "
        f"Recall={values[:,1].mean():.6f} "
        f"TV={values[:,2].mean():.6f}"
    )


print()
print("=== By SNR ===")

for snr in [-5.0, 0.0, 10.0]:

    values = np.array(
        results_by_snr[snr]
    )

    print(
        f"SNR={snr:g}: "
        f"MRR={values[:,0].mean():.6f} "
        f"Recall={values[:,1].mean():.6f} "
        f"TV={values[:,2].mean():.6f}"
    )


# TV for mixtures only
mixture_tv = []

for query_id, tv in zip(
    query_ids,
    tv_values,
):
    if metadata[str(query_id)]["k"] >= 2:
        mixture_tv.append(tv)


print()
print(
    "TV k>=2:",
    np.mean(mixture_tv),
)

print()
print("SUCCESS")
