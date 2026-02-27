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


def to_tokens(model, tokenizer, input_):
    return tokenizer(
        input_,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )["input_ids"]


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

    def _get_data_as_tensors(self, dataset: ConceptDataset, batch_size: int):
        """
        Converts data from the ConceptDataset into model-ready tensors.
        Assumes that the dataset yields (prompts, labels) and uses left padding.
        """
        data = []
        for batch in dataset.get_batches(batch_size=batch_size):
            prompts = batch["prompt"]
            tokens = to_tokens(self.model, self.model.tokenizer, prompts)
            data.append(tokens)
        return data

    def _get_module_name(self, layer_number, key_or_query):
        if key_or_query == "key":
            return key_module_name.format(layer_number)
        elif key_or_query == "query":
            return query_module_name.format(layer_number)
        else:
            raise ValueError(
                f"Invalid key_or_query: {key_or_query}. Must be 'key' or 'query'."
            )

    @torch.no_grad()
    def generate_multiple_layer_activations_and_freq(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        heads: List[Tuple[int, int]],
        batch_size: int = 16,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        For each sample in the dataset, returns the activations from multiple layers
        and a frequency vector corresponding to each non-padding token.

        The output for each layer is a tensor of shape:
            (num_tokens, d_model)
        where num_tokens is the total number of non-padding tokens across the dataset.
        The frequency vector (of shape (num_tokens,)) is built from a vocabulary frequency
        computed over the dataset.

        Args:
            dataset (ConceptDataset): Dataset yielding samples.
            heads (List[Tuple[int, int]]): List of layer numbers to extract activations from.
            batch_size (int): Batch size for processing the dataset.

        Returns:
            A tuple (final_activations, freq) where:
              - final_activations: List of tensors, one per layer, each of shape (num_tokens, d_model).
              - freq: Tensor of shape (num_tokens,), where each entry is the frequency of that token.
        """
        n_heads = self.model.config.num_attention_heads
        num_kv_groups = self.model.model.layers[0].self_attn.num_key_value_groups

        # Build the global vocabulary frequency dictionary.
        vocab_freq = self.build_vocab_frequency(dataset, batch_size=batch_size)

        data = self._get_data_as_tensors(dataset, batch_size)
        all_queries = [[] for _ in heads]
        all_keys = [[] for _ in heads]
        all_masks = []
        all_token_ids = []

        record_module_names = [
            self._get_module_name(layer, "key") for (layer, _) in heads
        ] + [self._get_module_name(layer, "query") for (layer, _) in heads]

        for batch in tqdm(data, desc="Generating multi-layer activations with freq"):
            if isinstance(batch, dict):
                inputs = {k: v.to(self.data_device) for k, v in batch.items()}
                input_ids = inputs["input_ids"]
            else:
                input_ids = batch.to(self.data_device)
                inputs = None

            # Run the model and obtain cache.
            with record_activations(self.model, record_module_names) as cache:
                self.model(input_ids)

            # Create mask for non-padding tokens.
            pad_token_id = self.model.tokenizer.pad_token_id
            mask = input_ids != pad_token_id
            all_masks.append(mask.cpu())

            # Extract non-padding token IDs.
            nonpad_ids = input_ids[mask.bool()].view(-1)
            all_token_ids.append(nonpad_ids.cpu())

            for idx, (layer, head_idx) in enumerate(heads):
                key_module = self._get_module_name(layer, "key")
                query_module = self._get_module_name(layer, "query")
                # Get activations: shape (batch_size, seq_len, d_model)
                # [batch, seq, d_head]
                keys = repeat_kv(cache[key_module][0], num_kv_groups)[
                    :, head_idx, :, :
                ].contiguous()
                # [batch, seq, d_head]
                queries = cache[query_module][0][:, head_idx, :, :].contiguous()
                # Immediately move activations to CPU.
                all_queries[idx].append(keys.cpu())
                all_keys[idx].append(queries.cpu())

            del cache
            torch.cuda.empty_cache()

        # Concatenate activations for each layer and token IDs.
        final_queries = [
            torch.cat(queries, dim=0) for queries in all_queries
        ]
        final_keys = [
            torch.cat(keys, dim=0) for keys in all_keys
        ]
        final_masks = [
            torch.cat(masks, dim=0) for masks in all_masks
        ]
        breakpoint()
        token_ids_all = torch.cat(all_token_ids, dim=0)
        # Build the frequency vector: for each token in token_ids_all, look up its global frequency.
        freq = torch.tensor([vocab_freq[token.item()] for token in token_ids_all])
        return final_activations, freq


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
        tokens = to_tokens(prompts, padding_side="left")

        input_ids = tokens.to(act_generator.data_device)
        pad_token_id = act_generator.model.tokenizer.pad_token_id
        bos_token_id = act_generator.model.tokenizer.bos_token_id
        attention_mask = (input_ids != pad_token_id) & (input_ids != bos_token_id)

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
        tokens = act_generator.model.to_tokens(prompts, padding_side="left")
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
