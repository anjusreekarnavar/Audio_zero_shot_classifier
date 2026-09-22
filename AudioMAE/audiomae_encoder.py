import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import models_mae


class AudioMAEEncoder:
    """
    Frozen AudioMAE encoder for the Esper reference-matching experiment.

    Output:
        L2-normalized 768-dimensional embedding.
    """

    MEAN = -4.2677393
    STD = 4.5689974
    MEL_BINS = 128
    TARGET_LENGTH = 1024

    def __init__(
        self,
        checkpoint="ckpt/pretrained.pth",
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Loading AudioMAE on {self.device}...")

        self.model = models_mae.mae_vit_base_patch16(
            audio_exp=True,
            img_size=(1024, 128),
            in_chans=1,
        )

        checkpoint_data = torch.load(
            checkpoint,
            map_location="cpu",
        )

        msg = self.model.load_state_dict(
            checkpoint_data["model"],
            strict=False,
        )

        print("Missing keys:", len(msg.missing_keys))
        print("Unexpected keys:", len(msg.unexpected_keys))

        self.model.to(self.device)
        self.model.eval()

        print("AudioMAE loaded.")

    def load_audio(self, wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)

        # Same preprocessing used by AudioMAE
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.MEL_BINS,
            dither=0.0,
            frame_shift=10,
        )

        n_frames = fbank.shape[0]
        padding = self.TARGET_LENGTH - n_frames

        if padding > 0:
            fbank = torch.nn.ZeroPad2d(
                (0, 0, 0, padding)
            )(fbank)

        elif padding < 0:
            fbank = fbank[:self.TARGET_LENGTH, :]

        # AudioSet pretrained normalization
        fbank = (fbank - self.MEAN) / (self.STD * 2)

        # [time, mel]
        #      ↓
        # [batch, channel, time, mel]
        return fbank.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def encode(self, wav_path):
        x = self.load_audio(wav_path)
        x = x.to(self.device)

        # [1, 513, 768]
        tokens = self.model.forward_encoder_no_mask(x)

        # Exclude CLS token and mean-pool the 512 patch tokens.
        #
        # This is OUR retrieval pooling choice, rather than
        # supervised AudioMAE classification.
        embedding = tokens[:, 1:, :].mean(dim=1)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
