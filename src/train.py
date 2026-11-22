import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

import config
from dataset import NASACellDataset, get_transforms

COL_HR = getattr(config, "COL_HR", "hr_post_exposure")


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def compute_stratified_metrics(val_df, group_col):
    grouped = (
        val_df.groupby(group_col)
        .agg(
            accuracy=("correct", "mean"),
            count=("correct", "size"),
        )
        .reset_index()
        .sort_values(by=group_col)
    )
    return grouped


def plot_stratified_accuracy(dose_stats, hr_stats, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panels = [
        (axes[0], dose_stats, config.COL_DOSE, "Accuracy by Radiation Dose (Gy)", "#2563eb"),
        (axes[1], hr_stats, COL_HR, "Accuracy by Post-Exposure Time (hr)", "#059669"),
    ]

    for ax, stats, col, title, color in panels:
        labels = [str(v) for v in stats[col]]
        accs = stats["accuracy"].values * 100
        counts = stats["count"].values

        bars = ax.bar(labels, accs, color=color, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, acc, n in zip(bars, accs, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{acc:.1f}%\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[0].set_ylabel("Validation Accuracy (%)")
    plt.suptitle("Stratified Validation Performance (Plate-Level Split)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    df = pd.read_csv(config.SUBSET_META_PATH)

    # Extract biological plate ID from filename to prevent slide-level data leakage
    df[config.COL_GROUP] = df[config.COL_FILENAME].str.split("-").str[0]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    train_idx, val_idx = next(splitter.split(df, df[config.COL_PARTICLE], groups=df[config.COL_GROUP]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print("Val class counts:\n", val_df[config.COL_PARTICLE].value_counts())

    train_loader = DataLoader(
        NASACellDataset(train_df, config.IMG_DIR, get_transforms(is_train=True)),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        NASACellDataset(val_df, config.IMG_DIR, get_transforms(is_train=False)),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1 = 0.0
    best_preds, best_targets = [], []

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(labels.tolist())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_preds, best_targets = all_preds, all_targets
            torch.save(model.state_dict(), config.MODEL_PATH)

    # Attach best predictions to validation metadata for stratified analysis
    val_df["true_label"] = best_targets
    val_df["pred_label"] = best_preds
    val_df["correct"] = (val_df["true_label"] == val_df["pred_label"]).astype(int)
    val_df.to_csv(config.OUTPUT_DIR / "val_split.csv", index=False)

    dose_stats = compute_stratified_metrics(val_df, config.COL_DOSE)
    hr_stats = compute_stratified_metrics(val_df, COL_HR)

    print("\n--- Validation Accuracy by Radiation Dose (dose_Gy) ---")
    print(dose_stats.to_string(index=False))

    print("\n--- Validation Accuracy by Post-Exposure Time (hr_post_exposure) ---")
    print(hr_stats.to_string(index=False))

    # Save metrics JSON
    metrics = {
        "best_val_f1": round(best_f1, 4),
        "best_val_acc": round(accuracy_score(best_targets, best_preds), 4),
        "by_dose_Gy": {
            str(r[config.COL_DOSE]): {"acc": round(float(r["accuracy"]), 4), "n": int(r["count"])}
            for _, r in dose_stats.iterrows()
        },
        "by_hr_post_exposure": {
            str(r[COL_HR]): {"acc": round(float(r["accuracy"]), 4), "n": int(r["count"])}
            for _, r in hr_stats.iterrows()
        },
    }
    with open(config.OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot confusion matrix
    cm = confusion_matrix(best_targets, best_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["X-ray", "Fe"])
    disp.plot(cmap="Blues")
    plt.title("Validation Confusion Matrix (Plate-Level Split)")
    plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot stratified accuracy bar charts
    plot_stratified_accuracy(dose_stats, hr_stats, config.OUTPUT_DIR / "stratified_accuracy.png")
    print(f"\nSaved model, metrics, and plots to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()