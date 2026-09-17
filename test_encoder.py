from src.encoder import AudioEncoder


wav = "/scratch/s5284664/ESC-50-master/audio/1-100032-A-0.wav"

encoder = AudioEncoder()

embedding = encoder.encode(wav)

print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])
print("Norm:", (embedding ** 2).sum() ** 0.5)