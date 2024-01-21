import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

import torchvision.models as models

from models.VanillaNet_LSLU import vanillanet_5 as create_model



def class_selectivity_index(weights):
    max_class_mean = weights.max(dim=1)[0]
    all_class_means = weights.mean(dim=1)
    selectivity_index = max_class_mean - all_class_means
    normalized_index = (selectivity_index - selectivity_index.min()) / (selectivity_index.max() - selectivity_index.min())
    return normalized_index

def analyze_model():
    model = create_model(num_classes=100)

    model_weight_path = "./checkpoint-best.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')
    model.load_state_dict(pre_weights, strict=False)

    model.eval()

    block_indices = []

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data
            index = class_selectivity_index(weights)
            block_indices.append(index.numpy())

    for i, indices in enumerate(block_indices, 1):
        plot_distribution(indices, f'Block {i}')

def plot_distribution(indices, title):
    sns.set(style="whitegrid", rc={"axes.labelsize": 8, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8})

    plt.figure(figsize=(8, 5))

    flattened_indices = indices.flatten()
    print(flattened_indices)
    sns.histplot(flattened_indices, kde=True, color='green', linewidth=2)

    plt.title(title)
    plt.xlabel('Class Selectivity Index')
    plt.ylabel('Density (PDF)')
    plt.show()

if __name__ == "__main__":
    analyze_model()
