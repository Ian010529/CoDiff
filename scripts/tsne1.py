import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.progress import track

from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model


def get_latents(model, batch):
    """
    Extracts latent vectors from the model before and after the LatentSelfAttention module.
    """
    # Move batch to the same device as the model
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)

    feats_ref = batch["motion"]
    lengths = batch["length"]
    
    with torch.no_grad():
        # VAE encoding
        if model.vae_type in ["mld", "vposert", "actor"]:
            z, _ = model.vae.encode(feats_ref, lengths)
        elif model.vae_type == "no":
            z = feats_ref.permute(1, 0, 2)
        else:
            raise TypeError(f"Unsupported vae_type: {model.vae_type}")

    z_before = z.clone()

    # Apply the latent enhancer if it exists
    if hasattr(model, 'latent_enhancer'):
        z_after = model.latent_enhancer(z)
    else:
        # If no enhancer, the latents are the same
        z_after = z.clone()

    # Squeeze the first dimension if it's 1, and handle different tensor shapes
    if z_before.shape[0] == 1:
        z_before = z_before.squeeze(0)
    if z_after.shape[0] == 1:
        z_after = z_after.squeeze(0)

    return z_before.cpu().numpy(), z_after.cpu().numpy(), batch['action'].cpu().numpy().flatten()


def plot_tsne(output_dir, latents_before, latents_after, labels, class_names):
    """
    Generates and saves a side-by-side t-SNE plot.
    """
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    
    # Fit and transform for both sets of latents
    tsne_before = tsne.fit_transform(latents_before)
    tsne_after = tsne.fit_transform(latents_after)
    
    df_before = pd.DataFrame({
        "comp-1": tsne_before[:, 0],
        "comp-2": tsne_before[:, 1],
        "label": [class_names[l] for l in labels]
    })
    
    df_after = pd.DataFrame({
        "comp-1": tsne_after[:, 0],
        "comp-2": tsne_after[:, 1],
        "label": [class_names[l] for l in labels]
    })
    
    print("Plotting results...")
    sns.set(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    sns.scatterplot(x="comp-1", y="comp-2", hue="label", data=df_before, ax=ax1, s=50, alpha=0.7)
    ax1.set_title("Before LatentSelfAttention")
    
    sns.scatterplot(x="comp-1", y="comp-2", hue="label", data=df_after, ax=ax2, s=50, alpha=0.7)
    ax2.set_title("After LatentSelfAttention")

    plt.suptitle("t-SNE Visualization of Latent Space", fontsize=16)
    
    # Save the figure
    save_path = Path(output_dir) / "tSNE_comparison.png"
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")
    plt.show()

def main():
    # Parse configuration
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER
    
    # Create output directory
    output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "tSNE_visualization_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration loaded:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    pl.seed_everything(cfg.SEED_VALUE)

    # GPU settings
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.DEVICE))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load dataset
    datasets = get_datasets(cfg, phase="test")
    if not datasets:
        print("No datasets found. Exiting.")
        return
        
    dataset = datasets[0]
    print(f"Dataset '{dataset.name}' loaded.")

    # Create model
    model = get_model(cfg, dataset)
    print(f"Model '{cfg.model.model_type}' loaded.")

    # Load checkpoint
    print(f"Loading checkpoint from {cfg.TEST.CHECKPOINTS}")
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Data loader
    dataloader = DataLoader(dataset.test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
    
    all_latents_before, all_latents_after, all_labels = [], [], []

    # Extract latents
    for batch in track(dataloader, description="Extracting latents..."):
        latents_before, latents_after, labels = get_latents(model, batch)
        all_latents_before.append(latents_before)
        all_latents_after.append(latents_after)
        all_labels.append(labels)

    all_latents_before = np.concatenate(all_latents_before, axis=0)
    all_latents_after = np.concatenate(all_latents_after, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Get class names if available
    class_names = {i: str(i) for i in range(dataset.nclasses)}
    if hasattr(dataset.test_dataset, '_action_classes'):
        class_names = dataset.test_dataset._action_classes

    # Plot t-SNE
    plot_tsne(output_dir, all_latents_before, all_latents_after, all_labels, class_names)

if __name__ == "__main__":
    main()