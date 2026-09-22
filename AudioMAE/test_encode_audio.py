import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import models_mae


CHECKPOINT = "ckpt/pretrained.pth"

MEAN = -4.2677393
STD = 4.5689974
MEL_BINS = 128
TARGET_LENGTH = 1024


def wav_to_fbank(filename):
    waveform, sr = torchaudio.load(filename)

    # AudioMAE preprocessing
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=MEL_BINS,
        dither=0.0,
        frame_shift=10,
    )

    # Pad/crop to 1024 frames.
    n_frames = fbank.shape[0]
    p = TARGET_LENGTH - n_frames

    if p > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LENGTH, :]

    # Same normalization as AudioMAE.
    fbank = (fbank - MEAN) / (STD * 2)

    # [1024,128] -> [1,1,1024,128]
    fbank = fbank.unsqueeze(0).unsqueeze(0)

    return fbank


print("Creating AudioMAE...")

model = models_mae.mae_vit_base_patch16(
    audio_exp=True,
    img_size=(1024, 128),
    in_chans=1,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

msg = model.load_state_dict(
    checkpoint["model"],
    strict=False,
)

model.eval()

print("Model loaded.")
print("Missing keys:", len(msg.missing_keys))
print("Unexpected keys:", len(msg.unexpected_keys))


# Pick one ESC-50 WAV automatically.
import glob
wav_path = (
    "/export/home/s5284664/esper-audio-challenge/"
    "field_queries/dev/dev_00000.wav"
)

print("Test WAV:", wav_path)

x = wav_to_fbank(wav_path)

print("Fbank shape:", x.shape)
print("Fbank min/max:", x.min().item(), x.max().item())


with torch.no_grad():

    tokens = model.forward_encoder_no_mask(x)

    print("Encoder output shape:", tokens.shape)

    # Average patch tokens.
    #
    # Token 0 is the CLS token.
    # Tokens 1: are the spectrogram patch tokens.
    embedding = tokens[:, 1:, :].mean(dim=1)

    # Cosine-similarity retrieval needs unit-length vectors.
    embedding = F.normalize(
        embedding,
        p=2,
        dim=-1,
    )


embedding = embedding.squeeze(0).cpu().numpy()

print("Embedding shape:", embedding.shape)
print("Embedding norm:", np.linalg.norm(embedding))

print("SUCCESS")
