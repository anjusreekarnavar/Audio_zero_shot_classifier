import numpy as np

from src.metrics import reciprocal_rank
from src.metrics import recall_at_k
from src.metrics import total_variation


# Imagine four classes:
#
# 0 = dog
# 1 = cat
# 2 = car_horn
# 3 = train

truth = np.array([
    0.7,   # dog
    0.0,   # cat
    0.3,   # car_horn
    0.0,   # train
])

prediction = np.array([
    0.6,   # dog
    0.1,   # cat
    0.2,   # car_horn
    0.1,   # train
])


print("Reciprocal Rank:", reciprocal_rank(prediction, truth))
print("Recall@2:", recall_at_k(prediction, truth, 2))
print("TV:", total_variation(prediction, truth))
