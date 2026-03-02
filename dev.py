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
from torch.utils.data import DataLoader, TensorDataset, random_split

from initializations.projected_knn_qk import ReservoirKMeans
from data_utils.concept_dataset import SupervisedConceptDataset
from llm_utils.activation_generator import ActivationGenerator
from llm_utils.qk_generator import QKGenerator, extract_token_ids_sample_ids_and_labels
from modeling.qk_mfa import QKMFA
from modeling.train_qk import train_nll


# %%


data_path = "./data/supervised.json"
# model_name = "gpt2-small"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
heads = [(16, 1)]
data_device = "cuda"
model_device = "cuda"


# Gather query, key vectors.
qk_generator = QKGenerator(
    model_name,
    model_device=model_device,
    data_device=data_device,
)
dataset_obj = SupervisedConceptDataset(data_path)


# %%

query_vecs, key_vecs, attn_weights = qk_generator.generate_query_key_vecs(
    dataset_obj, heads
)

# %%

q_all = query_vecs[0][0:250_000]
k_all = key_vecs[0][0:250_000]

full_ds = TensorDataset(q_all, k_all)

loader = DataLoader(
    full_ds,
    batch_size=128,
    shuffle=True,  # always shuffle your training set
    pin_memory=True,
)

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


# We use 500 centroids, this is an arbitrary number and you can reduce it to capture more broad subspaces or increase to produce more semantic covariances.
#
# Should run in 3-5 minutes. For shorter runtime, sample points as the centroids (second cell)

# %%


num_centroids = min(500, pool_size)
vocab_size = qk_generator.model.config.vocab_size

knn_q = ReservoirKMeans(
    num_centroids,
    pool_size=pool_size,
    query_or_key="query",
    vocab_size=vocab_size,
    device=model_device,
    proj_dim=32,
)
q_centroids = knn_q.fit(loader)

knn_k = ReservoirKMeans(
    num_centroids,
    pool_size=pool_size,
    query_or_key="key",
    vocab_size=vocab_size,
    device=model_device,
    proj_dim=32,
)
k_centroids = knn_k.fit(loader)


# %%


# random points
# N = query_vecs[0].shape[0]
# idx = torch.randperm(N, device=query_vecs[0].device)[
#    :num_centroids
# ]  # sample without replacement
# q_centroids = query_vecs[0][idx]
# k_centroids = key_vecs[0][idx]


# ### Training
#
# We train using Negative Log Likelihood. We provided an implementation of a very simple training loop.
# We use R = 10 (covariance dim), feel free to experiment with different values. It mostly depends on the intrinsic dimension of the data.
#
# We train for 10 epochs, which is sufficient for the follow up interpretation and steering. For evaluations, would want to train until convergence.
#
# Should take about 10-15 minutes. Feel free to train for less epochs, a couple epochs are often enough to see results (depends on dataset size).

# %%


model = QKMFA(q_centroids=q_centroids, k_centroids=k_centroids, rank=10).to(
    model_device
)
train_nll(model, loader, epochs=30, lr=1e-3)
