import torch
import models_mae

checkpoint_path = "ckpt/pretrained.pth"

print("Creating AudioMAE model...")

model = models_mae.mae_vit_base_patch16(
    audio_exp=True,
    img_size=(1024, 128),
    in_chans=1,
)

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
)

msg = model.load_state_dict(
    checkpoint["model"],
    strict=False,
)

print("\nMissing keys:")
print(msg.missing_keys)

print("\nUnexpected keys:")
print(msg.unexpected_keys)

print("\nSUCCESS: pretrained AudioMAE loaded")
print("Encoder dimension:", model.cls_token.shape[-1])
