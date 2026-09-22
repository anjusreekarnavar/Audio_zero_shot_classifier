import numpy as np

from audiomae_encoder import AudioMAEEncoder


wav_path = (
    "/export/home/s5284664/esper-audio-challenge/"
    "field_queries/dev/dev_00000.wav"
)

encoder = AudioMAEEncoder(
    checkpoint="ckpt/pretrained.pth"
)

embedding = encoder.encode(wav_path)

print()
print("WAV:", wav_path)
print("Embedding shape:", embedding.shape)
print("Embedding norm:", np.linalg.norm(embedding))

assert embedding.shape == (768,)
assert np.isclose(
    np.linalg.norm(embedding),
    1.0,
    atol=1e-5,
)

print("SUCCESS")
