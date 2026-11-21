import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

import config
from dataset import INV_LABEL_MAP, get_transforms
from train import build_model

def load_raw_rgb(img_path):
    img = tifffile.imread(img_path).astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # Resize to 224x224 for overlay alignment
    t = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
    t = transforms.Resize((224, 224), antialias=True)(t)
    return t.permute(1, 2, 0).numpy()  # (224, 224, 3) float in [0, 1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    val_df = pd.read_csv(config.OUTPUT_DIR / "val_split.csv")

    # Pick 3 X-ray and 3 Fe samples from validation set
    samples = pd.concat([
        val_df[val_df[config.COL_PARTICLE] == "X-ray"].head(3),
        val_df[val_df[config.COL_PARTICLE] == "Fe"].head(3),
    ]).reset_index(drop=True)

    val_tfms = get_transforms(is_train=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, row in samples.iterrows(): 
        img_path = config.IMG_DIR / row[config.COL_FILENAME]
        rgb_float = load_raw_rgb(img_path)

        # Prepare normalized model input
        input_tensor = val_tfms(torch.tensor(rgb_float).permute(2, 0, 1)).unsqueeze(0).to(device)
        pred_class = model(input_tensor).argmax(dim=1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets = None)[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        ax = axes[idx // 3, idx % 3]
        ax.imshow(overlay)
        ax.set_title(f"True: {row[config.COL_PARTICLE]} | Pred: {INV_LABEL_MAP[pred_class]}")
        ax.axis("off")

    plt.suptitle("Grad-CAM Attention on NASA BPS Cell Nuclei", fontsize=14)
    plt.tight_layout()
    out_path = config.OUTPUT_DIR / "gradcam_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Grad-CAM grid to {out_path}")

if __name__ == "__main__":
    main()