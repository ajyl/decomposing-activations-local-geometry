from typing import List, Tuple, Union, Callable
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv

from hook_utils import convert_to_hooked_model, record_activations
from data_utils.concept_dataset import ConceptDataset, SupervisedConceptDataset
from llm_utils.activation_generator import ActivationGenerator


key_module_name = "model.layers.{}.self_attn.hook_key_states"
value_module_name = "model.layers.{}.self_attn.hook_value_states"
query_module_name = "model.layers.{}.self_attn.hook_query_states"
attn_module_name = "model.layers.{}.self_attn.hook_attn_pattern"
qk_module_name = "model.layers.{}.self_attn.hook_qk_logits"
resid_mid_module_name = "model.layers.{}.hook_resid_mid"
resid_post_module_name = "model.layers.{}.hook_resid_post"


def to_tokens(tokenizer, input_):
    return tokenizer(
        input_,
        return_tensors="pt",
        padding=True,
        max_length=64,
    )


class QKGenerator(ActivationGenerator):
    def __init__(
        self,
        model_name: str,
        model_device: str = "cpu",
        data_device: str = "cpu",
        mode: str = "qk",
    ):
        """
        Initialize the generator with a pretrained model.

        Args:
            model_name (str): Name of the pretrained model.
            model_device (str): Device to load the model onto.
            data_device (str): Device to load the data onto.
            mode (str): Which activation to use ("mlp" or "residual").
        """
        super().__init__(model_name, model_device, data_device, mode, initialize=False)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_device,
            attn_implementation="eager",
        )
        self.model.eval()
        convert_to_hooked_model(self.model)
        self.model.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.tokenizer.padding_side = "left"

        self.model_name = model_name  # store for later use in helper functions
        self.data_device = data_device
        self._mode = mode

    def _tokenize_data(self, dataset: ConceptDataset, batch_size: int):
        """
        Converts data from the ConceptDataset into model-ready tensors.
        Assumes that the dataset yields (prompts, labels) and uses left padding.
        """
        data = []
        for batch in dataset.get_batches(batch_size=batch_size):
            prompts = batch["prompt"]
            tokenized = to_tokens(self.model.tokenizer, prompts)
            data.append(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                }
            )
        return data

    def _get_module_name(self, layer_number, module):
        if module == "key":
            return key_module_name.format(layer_number)
        elif module == "query":
            return query_module_name.format(layer_number)
        elif module == "attn_pattern":
            return attn_module_name.format(layer_number)
        else:
            raise ValueError(f"Invalid module {module}.")

    @torch.no_grad()
    def generate_query_key_vecs(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        heads: List[Tuple[int, int]],
        batch_size: int = 16,
        top_k: int = 8,
        attn_min: float = 0.01,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        For each sample in the dataset, returns the queries and keys from specified heads.

        Args:
            dataset (ConceptDataset): Dataset yielding samples.
            heads (List[Tuple[int, int]]): List of head indices to collect query, keys from.
            batch_size (int): Batch size for processing the dataset.

        Returns:
            A tuple (query_vecs, key_vecs).
        """
        self.model.eval()
        d_head = self.model.config.head_dim
        n_heads = self.model.config.num_attention_heads
        num_kv_groups = self.model.model.layers[0].self_attn.num_key_value_groups

        data = self._tokenize_data(dataset, batch_size)
        all_queries = [[] for _ in heads]
        all_keys = [[] for _ in heads]
        all_weights = [[] for _ in heads]
        all_masks = []

        layers = sorted(set(layer for layer, _ in heads))
        record_module_names = (
            [self._get_module_name(layer, "key") for layer in layers]
            + [self._get_module_name(layer, "query") for layer in layers]
            + [self._get_module_name(layer, "attn_pattern") for layer in layers]
        )

        for batch in tqdm(data, desc="Gathering query, key vectors..."):
            input_ids = batch["input_ids"].to(self.data_device)
            attention_mask = batch["attention_mask"].to(self.data_device)

            # Run the model and obtain cache.
            with record_activations(self.model, record_module_names) as cache:
                self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract non-padding token IDs.
            # nonpad_ids = input_ids[mask.bool()].view(-1)
            mask = attention_mask.bool()
            B, seq = input_ids.shape
            pos_q = (
                torch.arange(seq, device=input_ids.device)
                .view(1, seq, 1)
                .expand(B, seq, top_k)
            )  # (B, seq, top_k)

            for idx, (layer, head_idx) in enumerate(heads):
                query_module = self._get_module_name(layer, "query")
                key_module = self._get_module_name(layer, "key")
                attn_module = self._get_module_name(layer, "attn_pattern")

                # [batch, seq, d_head]
                queries = cache[query_module][0][:, head_idx, :, :].contiguous()
                keys = repeat_kv(cache[key_module][0], num_kv_groups)[
                    :, head_idx, :, :
                ].contiguous()
                attn_pattern = cache[attn_module][0][
                    :, head_idx, :, :
                ].contiguous()  # [batch, seq, seq]

                attn_masked = attn_pattern.masked_fill(
                    ~mask[:, None, :].bool(), float("-inf")
                )

                _top_k = min(top_k, seq)
                top_v, top_i = torch.topk(attn_masked, k=_top_k, dim=-1)

                pos_q_h = pos_q
                if _top_k != top_k:
                    # shrink the precomputed grid view for this head only
                    pos_q_h = (
                        torch.arange(seq, device=input_ids.device)
                        .view(1, seq, 1)
                        .expand(B, seq, _top_k)
                    )

                causal_mask = top_i <= pos_q_h

                # Key validity mask at selected indices
                # mask_keys[b,t,j] = mask[b, top_i[b,t,j]]
                # [B, seq, top_k] gather from [B, seq] using indices [B, seq, top_k]
                mask_keys = mask.gather(dim=1, index=top_i.reshape(B, -1)).reshape(
                    B, seq, _top_k
                )
                # Should be all 1s, i.e., all selected keys are valid (non-padding)
                # assert mask_keys.all().item()

                mask_queries = mask[:, :, None].expand(B, seq, _top_k)

                # Edge validity + optional attention threshold
                # final_mask[b,t,j] = mask[b,t] & mask[b, top_i[b,t,j]] & causal_mask[b,t,j] & (top_v[b,t,j] > attn_min)
                final_mask = mask_queries & mask_keys & causal_mask & (top_v > attn_min)
                if not final_mask.any():
                    breakpoint()

                # Gather key vectors for selected key positions
                # k: (B,seq,d), top_i: (B,seq,_top_k)
                k_gather = keys.gather(
                    dim=1,
                    index=top_i.reshape(B, seq * _top_k)
                    .unsqueeze(-1)
                    .expand(B, seq * _top_k, d_head),
                ).reshape(
                    B, seq, _top_k, d_head
                )  # (B,seq,_top_k,d)

                # Repeat queries to align with (t, selected s)
                q_rep = queries.unsqueeze(2).expand(
                    B, seq, _top_k, d_head
                )  # (B,seq,_top_k,d)

                q_vecs = q_rep[final_mask].cpu()
                k_vecs = k_gather[final_mask].cpu()
                attn_weights = top_v[final_mask].cpu()

                all_queries[idx].append(q_vecs)
                all_keys[idx].append(k_vecs)
                all_weights[idx].append(attn_weights)

            # del cache
            # torch.cuda.empty_cache()

        final_queries = [torch.cat(xs, dim=0) for xs in all_queries]
        final_keys = [torch.cat(xs, dim=0) for xs in all_keys]
        final_weights = [torch.cat(xs, dim=0) for xs in all_weights]
        return final_queries, final_keys, final_weights


def extract_token_ids_sample_ids_and_labels(
    dataset: ConceptDataset, act_generator: ActivationGenerator, batch_size: int = 5
):
    """
    Efficiently extract non-padding token IDs and corresponding labels from a dataset using the provided
    act_generator's tokenizer (without running the model or extracting activations).

    Args:
        dataset (ConceptDataset): A dataset instance that yields batches with at least a "prompt" key.
        act_generator (ActivationGenerator): Instance with a model containing a tokenizer and data_device.
        batch_size (int): Batch size for processing the dataset.

    Returns:
        token_ids (torch.Tensor): Tensor of shape (num_tokens,) containing the token IDs
                                  for all non-padding tokens in the dataset.
        labels (List): List of labels corresponding to each non-padding token.
    """
    all_token_ids = []
    all_labels = []
    sample_ids = []
    pad_token_id = act_generator.model.tokenizer.pad_token_id
    idx = 0

    for batch in tqdm(
        dataset.get_batches(batch_size=batch_size), desc="Extracting token IDs"
    ):
        prompts = batch["prompt"]
        labels = batch["label"]

        # Tokenize the prompts (using left padding to be consistent)
        tokenized = to_tokens(act_generator.model.tokenizer, prompts)

        input_ids = tokenized["input_ids"].to(act_generator.data_device)
        attention_mask = tokenized["attention_mask"].to(act_generator.data_device)
        pad_token_id = act_generator.model.tokenizer.pad_token_id
        bos_token_id = act_generator.model.tokenizer.bos_token_id

        # Count non-padding tokens per sample and repeat labels accordingly
        num_non_padding = attention_mask.sum(dim=1).squeeze()
        for n, label in zip(num_non_padding, labels):
            all_labels += [label] * n
            sample_ids += [idx] * n
            idx += 1

        # Filter out pad tokens and collect token IDs
        nonpad_ids = input_ids[attention_mask].view(-1)
        all_token_ids.append(nonpad_ids.cpu())

    token_ids = torch.cat(all_token_ids, dim=0)
    return token_ids, sample_ids, all_labels


import torch
from tqdm import tqdm


def extract_token_ids_and_sample_ids(
    dataset: ConceptDataset, act_generator: ActivationGenerator, batch_size: int = 5
):
    """
    Efficiently extract non-padding token IDs and sample IDs from a dataset using the provided
    act_generator's tokenizer (without running the model or extracting activations).

    Args:
        dataset (ConceptDataset): A dataset instance that yields batches with at least a "prompt" key.
        act_generator (ActivationGenerator): Instance with a model containing a tokenizer and data_device.
        batch_size (int): Batch size for processing the dataset.

    Returns:
        token_ids (torch.Tensor): Tensor of shape (num_tokens,) containing the token IDs
                                  for all non-padding tokens in the dataset.
        sample_ids (List[int]): List of sample IDs corresponding to each non-padding token.
    """
    all_token_ids = []
    sample_ids = []
    pad_token_id = act_generator.model.tokenizer.pad_token_id
    bos_token_id = act_generator.model.tokenizer.bos_token_id
    idx = 0

    for batch in tqdm(
        dataset.get_batches(batch_size=batch_size), desc="Extracting token IDs"
    ):
        prompts = batch["prompt"]

        # Tokenize the prompts (using left padding to be consistent)
        tokens = to_tokens(act_generator.model.tokenizer, prompts)
        input_ids = tokens.to(act_generator.data_device)

        # Create attention mask ignoring PAD and BOS
        attention_mask = (input_ids != pad_token_id) & (input_ids != bos_token_id)

        # Count non-padding tokens per sample and repeat sample IDs accordingly
        num_non_padding = attention_mask.sum(dim=1).squeeze()
        for n in num_non_padding:
            sample_ids += [idx] * n
            idx += 1

        # Filter out pad tokens and collect token IDs
        nonpad_ids = input_ids[attention_mask].view(-1)
        all_token_ids.append(nonpad_ids.cpu())

    token_ids = torch.cat(all_token_ids, dim=0)
    return token_ids, sample_ids
