# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from __future__ import annotations
import os
import torch
from llm_utils.activation_generator import ActivationGenerator
from llm_utils.qk_generator import QKGenerator
from data_utils.concept_dataset import SupervisedConceptDataset
from torch.utils.data import DataLoader, TensorDataset, random_split


# ## Configuration
#
# - `data_path`: Path to the dataset / examples used for training and analysis.
# - `model_name`: A **TransformerLens-supported** model to extract activations from. We default to a small model (`gpt2-small`) for fast iteration; swap in larger models if you have GPU memory/compute.
# - `layers`: Which layer to inspect and factorize.
# - `data_device`: Where data tensors live during preprocessing (CPU by default).
# - `model_device`: Where the model runs for activation extraction and generation. Use `mps` on Apple Silicon, `cuda` on NVIDIA GPUs, or `cpu` if needed.
# - `factorization_mode`: Which activation stream to factorize:
#   - `residual`: a general-purpose choice that often yields clean, interpretable structure.
#

# %%


data_path = "./data/supervised_small.json"
#model_name = "gpt2-small"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
layers = [16]
data_device = "cuda"
model_device = "cuda"
factorization_mode = "residual"


# ### Loading and Generating Data
#
# In this tutorial we use our own abstractions for generating activations and loading data. In the end, you need to generate a loader for training MFA. Feel free to swap out with a different method.

# %%


act_generator = ActivationGenerator(
    model_name,
    model_device=model_device,
    data_device=data_device,
    mode=factorization_mode,
    initialize=True,
)


# %%


dataset_obj = SupervisedConceptDataset(data_path)


# %%


activations, _ = act_generator.generate_multiple_layer_activations_and_freq(
    dataset_obj, layers
)


# We additionally load tokens in order to later interpret the subspaces.

# %%


from llm_utils.activation_generator import extract_token_ids_sample_ids_and_labels

tokens, _, _ = extract_token_ids_sample_ids_and_labels(dataset_obj, act_generator)


# Creating the loaders from extracted activations
# To make this notebook work on lower compute we utilize only 250k activations.

# %%


# your raw data
data_size = 1000
X_all = activations[0][0:data_size]
tokens = tokens[0:data_size]

# make a single dataset
full_ds = TensorDataset(X_all, tokens)

loader = DataLoader(
    full_ds,
    batch_size=128,
    shuffle=True,  # always shuffle your training set
    pin_memory=True,
)

# if you still want a standalone token loader (e.g. for some other pass):
token_loader = DataLoader(tokens, batch_size=128)


# ### Initialization
#
# As described in the paper we tested three options for initialization.
# We found that K-Means often works well, with random point initialization also successful (random weights often fail).
# In this tutorial we show how to use K-Means as its the most complicated of the three, and we provide an implementation that works on torch.

# We must decide on how much of the data to run our K-Means. Since K-Means is slower, our implmentation allows to decide a pool size which will be randomly sampled. Additionally, for efficiency it uses a projected K-Means.
#
# In this tutorial we use the 20% dataset which consists of 600k activations in order to speed it up.

# %%


pool_size = round(len(loader.dataset) / 5)
pool_size


# We use 500 centroids, this is an arbitrary number and you can reduce it to capture more broad subspaces or increase to produce more semantic covariances.
#
# Should run in 3-5 minutes. For shorter runtime, sample points as the centroids (second cell)

# %%


from initializations.projected_knn import ReservoirKMeans

#num_centroids = 500
num_centroids = 10

knn = ReservoirKMeans(
    num_centroids,
    pool_size=pool_size,
    vocab_size=50257,
    device=model_device,
    proj_dim=32,
)
centroids = knn.fit(loader)


# %%


# random points
N = X_all.shape[0]
idx = torch.randperm(N, device=X_all.device)[
    :num_centroids
]  # sample without replacement
centroids = X_all[idx]


# ### Training
#
# We train using Negative Log Likelihood. We provided an implementation of a very simple training loop.
# We use R = 10 (covariance dim), feel free to experiment with different values. It mostly depends on the intrinsic dimension of the data.
#
# We train for 10 epochs, which is sufficient for the follow up interpretation and steering. For evaluations, would want to train until convergence.
#
# Should take about 10-15 minutes. Feel free to train for less epochs, a couple epochs are often enough to see results (depends on dataset size).

# %%


from modeling.mfa import MFA
from modeling.train import train_nll
breakpoint()

model = MFA(centroids=centroids, rank=10).to(model_device)
train_nll(model, loader, epochs=10, lr=1e-3)


