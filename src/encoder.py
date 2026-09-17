import wave

import numpy as np
import torch
import torch.nn.functional as F

from transformers import ClapModel, ClapProcessor


class AudioEncoder:
    def __init__(
        self,
        model_name="laion/clap-htsat-unfused",
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Loading CLAP on {self.device}...")

        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print("CLAP loaded.")

    def load_audio(self, wav_path):
        """
        Read an ESC-50 WAV using Python's built-in wave module
        and resample from 44.1 kHz to 48 kHz.
        """

        with wave.open(wav_path, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()

            if channels != 1:
                raise ValueError(
                    f"Expected mono audio, got {channels} channels"
                )

            if sample_width != 2:
                raise ValueError(
                    f"Expected 16-bit audio, got {sample_width * 8}-bit"
                )

            raw = wav.readframes(n_frames)

        # int16 PCM -> float32 [-1, 1]
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        audio /= 32768.0

        # CLAP expects 48 kHz.
        if sample_rate != 48000:
            audio_tensor = torch.from_numpy(audio)[None, None, :]

            new_length = round(
                audio.shape[0] * 48000 / sample_rate
            )

            audio_tensor = F.interpolate(
                audio_tensor,
                size=new_length,
                mode="linear",
                align_corners=False,
            )

            audio = (
                audio_tensor
                .squeeze(0)
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )

        return audio

    @torch.no_grad()
    def encode(self, wav_path):
        """
        Convert one WAV file into a normalized CLAP embedding.
        """

        audio = self.load_audio(wav_path)

        inputs = self.processor(
            audios=audio,
            sampling_rate=48000,
            return_tensors="pt",
        )

        input_features = inputs["input_features"].to(self.device)

        embedding = self.model.get_audio_features(
            input_features=input_features
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding.squeeze(0).cpu().numpy()
