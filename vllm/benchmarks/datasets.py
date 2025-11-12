# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena
"""

import argparse
import ast
import logging
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase

from vllm.benchmarks.placeholder import PlaceholderModule

try:
    from datasets import load_dataset
except ImportError:
    datasets = PlaceholderModule("datasets")
    load_dataset = datasets.placeholder_attr("load_dataset")

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")

try:
    from vllm import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: str | list[str]
    prompt_len: int
    expected_output_len: int
    lora_request: None = None
    request_id: str | None = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.

        Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
                indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
                sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.data = None

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
    ) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
                for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.
            request_id_prefix (str): The prefix of request_id.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
    ) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
                requests.
            num_requests (int): The target number of requests.
            request_id_prefix (str): The prefix applied to generated request
                identifiers.

        """
        if no_oversample:
            logger.info("Skipping oversampling. Total samples: %d.", len(requests))
            return

        if len(requests) < num_requests:
            random.seed(self.random_seed)
            needed = num_requests - len(requests)
            additional = []
            for i in range(needed):
                req = deepcopy(random.choice(requests))
                req.request_id = request_id_prefix + str(len(requests) + i)
                additional.append(req)
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)

        ids = [req.request_id for req in requests]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate request_id found in the sampled "
                "requests. Please ensure that each request_id "
                "is unique."
            )


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (
        prompt_too_short or output_too_short or prompt_too_long or combined_too_long
    )

def gen_prompt_decode_to_target_len(
    tokenizer: PreTrainedTokenizerBase,
    token_sequence: list[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[int]]:
    """
    Ensure decoded-then-encoded prompt length matches the target token length.

    This function decodes an initial token sequence to text and re-encodes it
    , iteratively adjusting the token sequence length to match a target.
    This is necessary because some tokenizers do not guarantee a 1:1 mapping
    between consecutive tokens and the decoded-then-encoded sequence length.
    For example, for GPT2Tokenizer:
    [6880, 6881] -> ['Ġcalls', 'here'] ->
    [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']

    Returns a tuple of the final prompt string and the adjusted token sequence.
    """
    remain_num_try = max_retry
    token_mismatch = 0
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if remain_num_try <= 0:
            if len(token_sequence) != target_token_len:
                token_mismatch = len(token_sequence) - target_token_len
            break

        if len(token_sequence) == target_token_len:
            break
        elif len(token_sequence) < target_token_len:
            if rng is not None:
                extra_tokens = rng.integers(
                    0,
                    tokenizer.vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            else:
                extra_tokens = np.random.randint(
                    0,
                    tokenizer.vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            token_sequence.extend(extra_tokens)
        elif len(token_sequence) > target_token_len:
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1

    return prompt, token_sequence, token_mismatch


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    """
    Synthetic text-only dataset for serving/throughput benchmarks.

    Strategy:
    - Sample input/output token lengths per request from integer-uniform ranges
      around configured means (controlled by range_ratio).
    - Prepend a fixed random prefix of length prefix_len.
    - Generate the remaining tokens as a reproducible sequence:
      (offset + index + arange(input_len)) % vocab_size.
    - Decode then re-encode/truncate to ensure prompt token counts match.
    - Uses numpy.default_rng seeded with random_seed for reproducible sampling.
    """

    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Use numpy's default_rng for deterministic sampling
        # Do not use random.seed() or np.random.seed() elsewhere in this class.
        # This ensures that the RNG is isolated from global RNG state.
        self._rng = np.random.default_rng(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        **kwargs,
    ) -> list[SampleRequest]:
        # validate total input tokens (prefix + sampled) is at least 1.
        num_special = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special)
        min_sampled_input = math.floor(real_input_len * (1.0 - float(range_ratio)))
        min_total_input = int(prefix_len) + min_sampled_input
        if min_total_input < 1:
            raise ValueError(
                "--random-input-len is too small: with tokenizer special "
                f"tokens {num_special} and --random-range-ratio {range_ratio}, "
                "the minimum possible total input tokens (prefix + sampled) is "
                f"{min_total_input}. Increase --random-input-len and/or "
                "--random-prefix-len, or decrease --random-range-ratio so that "
                "prefix_len + floor(max(0, random_input_len - num_special)) "
                "* (1 - range_ratio) >= 1."
            )

        input_lens, output_lens, offsets = self.get_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        vocab_size = tokenizer.vocab_size
        prohibited_tokens = tokenizer.all_special_ids
        all_tokens = np.arange(vocab_size)
        allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))

        # Generate prefix once
        prefix_token_ids = self.get_prefix(allowed_tokens, prefix_len)

        requests = []
        token_mismatch_total = 0
        for i in range(num_requests):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(  # noqa: E501
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
                allowed_tokens=allowed_tokens,
            )
            token_mismatch_total += token_mismatch
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    request_id=request_id_prefix + str(i),
                )
            )
        # only used for embeddings benchmark.
        if batchsize > 1:
            batch_requests = []
            # Create batched requests
            for i in range(0, num_requests, batchsize):
                batch = requests[i : i + batchsize]
                batch_requests.append(
                    SampleRequest(
                        prompt=[req.prompt for req in batch],
                        prompt_len=sum(req.prompt_len for req in batch),
                        expected_output_len=0,
                        request_id=request_id_prefix + str(i // batchsize),
                    )
                )
            requests = batch_requests

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )

        return requests

    def get_prefix(
        self,
        allowed_tokens: np.ndarray,
        prefix_len: int,
    ) -> list[int]:
        """
        Get the prefix for the dataset.
        """
        return (
            allowed_tokens[
                self._rng.integers(0, len(allowed_tokens), size=prefix_len)
            ].tolist()
            if prefix_len > 0
            else []
        )

    def get_sampling_params(
        self,
        num_requests: int,
        range_ratio: float,
        input_len: int,
        output_len: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the sampling parameters for the dataset.
        """
        # Enforce range_ratio < 1
        if not (0.0 <= range_ratio < 1.0):
            raise ValueError("range_ratio must be in [0, 1).")
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special_tokens)
        # Bounds use floor for low and ceil for high
        input_low = math.floor(real_input_len * (1 - range_ratio))
        input_high = math.ceil(real_input_len * (1 + range_ratio))
        output_low = math.floor(output_len * (1 - range_ratio))
        output_high = math.ceil(output_len * (1 + range_ratio))
        # Ensure the lower bound for output length is at least 1 to
        # prevent sampling 0 tokens.
        output_low = max(output_low, 1)
        output_high = max(output_high, 1)

        if input_low > input_high:
            raise ValueError(
                f"Invalid input sampling interval: low={input_low} > high={input_high}"
            )
        if output_low > output_high:
            raise ValueError(
                "Invalid output sampling interval: "
                f"low={output_low} > high={output_high}"
            )

        logger.info(
            "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
            input_low,
            input_high,
            output_low,
            output_high,
        )

        input_lens = self._rng.integers(input_low, input_high + 1, size=num_requests)
        output_lens = self._rng.integers(output_low, output_high + 1, size=num_requests)
        offsets = self._rng.integers(0, tokenizer.vocab_size, size=num_requests)
        return input_lens, output_lens, offsets

    def generate_token_sequence(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        prefix_token_ids: list[int],
        prefix_len: int,
        vocab_size: int,
        input_len: int,
        offset: int,
        index: int,
        allowed_tokens: np.ndarray,
    ) -> tuple[str, int, int]:
        """
        Returns (prompt, total_input_len).

        NOTE: After decoding the prompt we have to encode and decode it again.
        This is done because in some cases N consecutive tokens
        give a string tokenized into != N number of tokens.
        For example for GPT2Tokenizer:
        [6880, 6881] -> ['Ġcalls', 'here'] ->
        [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
        To avoid uncontrolled change of the prompt length,
        the encoded sequence is truncated before being decoded again.
        """
        # Build the inner sequence by sampling
        # sequentially from the allowed tokens
        inner_seq = allowed_tokens[
            (offset + index + np.arange(input_len)) % len(allowed_tokens)
        ].tolist()
        token_sequence = prefix_token_ids + inner_seq

        # Decode, then re-encode and truncate to preserve token count invariants
        total_input_len = prefix_len + int(input_len)
        prompt, adjusted_token_sequence, token_mismatch = (
            gen_prompt_decode_to_target_len(
                tokenizer=tokenizer,
                token_sequence=token_sequence,
                target_token_len=total_input_len,
                add_special_tokens=False,
                rng=self._rng,
            )
        )
        total_input_len = len(adjusted_token_sequence)
        return prompt, total_input_len, token_mismatch


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDatasetForReranking(RandomDataset):
    """
    Random dataset specialized for the needs of scoring:
    - Batches of inputs
    - Inputs composed of pairs
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        range_ratio: float = RandomDataset.DEFAULT_RANGE_RATIO,
        input_len: int = RandomDataset.DEFAULT_INPUT_LEN,
        batchsize: int = 1,
        is_reranker: bool = True,
        **kwargs,
    ) -> list[SampleRequest]:
        n_sep_tokens = int(is_reranker)

        query_len_param = (input_len // 2) - n_sep_tokens if is_reranker else input_len

        query_lens, _, query_offsets = self.get_sampling_params(
            1, range_ratio, query_len_param, 0, tokenizer
        )

        query_len = int(query_lens[0])

        if not is_reranker:
            assert num_requests > 1 and batchsize > 1
            num_requests -= 1
            batchsize -= 1
            doc_len_param = input_len
        else:
            doc_len_param = input_len - query_len - n_sep_tokens

        doc_lens, _, doc_offsets = self.get_sampling_params(
            num_requests, range_ratio, doc_len_param, 0, tokenizer
        )
        vocab_size = tokenizer.vocab_size

        query_prompt, query_input_len, token_mismatch_total = (
            self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=[],
                prefix_len=0,
                vocab_size=vocab_size,
                input_len=query_len,
                offset=int(query_offsets[0]),
                index=0,
            )
        )

        requests = []
        for i in range(num_requests):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(  # noqa: E501
                tokenizer=tokenizer,
                prefix_token_ids=[],
                prefix_len=0,
                vocab_size=vocab_size,
                input_len=int(doc_lens[i]),
                offset=int(doc_offsets[i]),
                index=i + 1,
            )
            token_mismatch_total += token_mismatch
            requests.append((prompt, total_input_len))

        batch_requests = []
        # Create batched requests
        for i in range(0, num_requests, batchsize):
            batch = requests[i : i + batchsize]
            query_contrib = (
                (query_input_len + n_sep_tokens) * len(batch)
                if is_reranker
                else query_input_len
            )
            batch_requests.append(
                SampleRequest(
                    prompt=[query_prompt] + [req[0] for req in batch],
                    prompt_len=query_contrib + sum(req[1] for req in batch),
                    expected_output_len=0,
                    request_id=request_id_prefix + str(i // batchsize),
                )
            )

        if token_mismatch_total != 0:
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                "more" if token_mismatch_total > 0 else "fewer",
            )

        return batch_requests

class _ValidateDatasetArgs(argparse.Action):
    """Argparse action to validate dataset name and path compatibility."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        # Get current values of both dataset_name and dataset_path
        dataset_name = getattr(namespace, "dataset_name", "random")
        dataset_path = getattr(namespace, "dataset_path", None)

        # Validate the combination
        if dataset_name == "random" and dataset_path is not None:
            parser.error(
                "Cannot use 'random' dataset with --dataset-path. "
                "Please specify the appropriate --dataset-name (e.g., "
                "'sharegpt', 'custom', 'sonnet') for your dataset file: "
                f"{dataset_path}"
            )


def add_dataset_parser(parser: FlexibleArgumentParser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        action=_ValidateDatasetArgs,
        choices=[
            "sharegpt",
            "burstgpt",
            "sonnet",
            "random",
            "random-mm",
            "random-rerank",
            "hf",
            "custom",
            "prefix_repetition",
            "spec_bench",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        action=_ValidateDatasetArgs,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Do not oversample if the dataset has fewer samples than num-prompts.",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Skip applying chat template to prompt for datasets that support it.",
    )
    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable shuffling of dataset samples for deterministic ordering.",
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help="Number of output tokens per request, used only for custom dataset.",
    )

    spec_bench_group = parser.add_argument_group("spec bench dataset options")
    spec_bench_group.add_argument(
        "--spec-bench-output-len",
        type=int,
        default=256,
        help="Num of output tokens per request, used only for spec bench dataset.",
    )
    spec_bench_group.add_argument(
        "--spec-bench-category",
        type=str,
        default=None,
        help="Category for spec bench dataset. If None, use all categories.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    blazedit_group = parser.add_argument_group("blazedit dataset options")
    blazedit_group.add_argument(
        "--blazedit-min-distance",
        type=float,
        default=0.0,
        help="Minimum distance for blazedit dataset. Min: 0, Max: 1.0",
    )
    blazedit_group.add_argument(
        "--blazedit-max-distance",
        type=float,
        default=1.0,
        help="Maximum distance for blazedit dataset. Min: 0, Max: 1.0",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )
    random_group.add_argument(
        "--random-batch-size",
        type=int,
        default=1,
        help=("Batch size for random sampling. Only used for embeddings benchmark."),
    )
    random_group.add_argument(
        "--no-reranker",
        action="store_true",
        help=(
            "Whether the model supports reranking natively."
            " Only used for reranker benchmark."
        ),
    )

    def _parse_mm_bucket_config(v: object) -> dict[tuple[int, int, int], float]:
        # If already a dict (e.g., programmatic call), normalize keys
        def normalize(d: dict) -> dict[tuple[int, int, int], float]:
            out: dict[tuple[int, int, int], float] = {}
            for k, val in d.items():
                key = k
                if isinstance(key, str):
                    with suppress(Exception):
                        key = ast.literal_eval(key)
                if not (
                    isinstance(key, tuple)
                    and len(key) == 3
                    and all(isinstance(x, int) for x in key)
                ):
                    raise ValueError(
                        f"Invalid bucket key {k!r}. Expected tuple (H, W, T)."
                    )
                out[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
            return out

        if isinstance(v, dict):
            return normalize(v)
        if isinstance(v, str):
            # Python literal (supports tuple keys)
            parsed = ast.literal_eval(v)
            if not isinstance(parsed, dict):
                raise ValueError("Bucket config must parse to a dict.")
            return normalize(parsed)
        raise ValueError("Unsupported value for --random-mm-bucket-config.")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-name",
        type=str,
        default=None,
        help=(
            "Name of the dataset on HuggingFace "
            "(e.g., 'lmarena-ai/VisionArena-Chat'). "
            "Specify this if your dataset-path is a local path."
        ),
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    prefix_repetition_group = parser.add_argument_group(
        "prefix repetition dataset options"
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=256,
        help="Number of prefix tokens per request, used only for prefix "
        "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=256,
        help="Number of suffix tokens per request, used only for prefix "
        "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=10,
        help="Number of prefixes to generate, used only for prefix repetition "
        "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for prefix "
        "repetition dataset.",
    )


def get_samples(args, tokenizer) -> list[SampleRequest]:
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    if args.dataset_name == "custom":
        dataset = CustomDataset(
            dataset_path=args.dataset_path, disable_shuffle=args.disable_shuffle
        )
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.skip_chat_template,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(
            dataset_path=args.dataset_path, disable_shuffle=args.disable_shuffle
        )
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        hf_kwargs = {}
        if (
            args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in InstructCoderDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MTBenchDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
            )

        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
            hf_name=args.hf_name,
            disable_shuffle=args.disable_shuffle,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            skip_chat_template=args.skip_chat_template,
            **hf_kwargs,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "spec_bench": lambda: SpecBench(
                dataset_path=args.dataset_path,
                category=args.spec_bench_category,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_len=args.spec_bench_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "random": lambda: RandomDataset(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
                request_id_prefix=args.request_id_prefix,
                batchsize=args.random_batch_size,
                no_oversample=args.no_oversample,
            ),
            "random-rerank": lambda: RandomDatasetForReranking(
                random_seed=args.seed,
                dataset_path=args.dataset_path,
                disable_shuffle=args.disable_shuffle,
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                input_len=args.random_input_len,
                range_ratio=args.random_range_ratio,
                request_id_prefix=args.request_id_prefix,
                batchsize=args.random_batch_size,
                is_reranker=not args.no_reranker,
            ),
        }

        try:
            # Enforce endpoint compatibility for multimodal datasets.
            if args.dataset_name == "random-mm" and args.backend not in ["openai-chat"]:
                raise ValueError(
                    "Multi-modal content (images) is only supported on "
                    "'openai-chat' backend."
                )
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests


# -----------------------------------------------------------------------------
# Custom Dataset Implementation
# -----------------------------------------------------------------------------


class CustomDataset(BenchmarkDataset):
    """
    Implements the Custom dataset.  Loads data from a JSONL file and generates
    sample requests based on conversation turns. E.g.,
    ```
    {"prompt": "What is the capital of India?"}
    {"prompt": "What is the capital of Iran?"}
    {"prompt": "What is the capital of China?"}
    ```
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        # self.data will be a list of dictionaries
        # e.g., [{"prompt": "What is the capital of India?"}, ...]
        # This will be the standardized format which load_data()
        # has to convert into depending on the filetype of dataset_path.
        # sample() will assume this standardized format of self.data
        self.data = []

        # Load the JSONL file
        if self.dataset_path.endswith(".jsonl"):
            jsonl_data = pd.read_json(path_or_buf=self.dataset_path, lines=True)

            # check if the JSONL file has a 'prompt' column
            if "prompt" not in jsonl_data.columns:
                raise ValueError("JSONL file must contain a 'prompt' column.")

            # Convert each row to a dictionary and append to self.data
            # This will convert the DataFrame to a list of dictionaries
            # where each dictionary corresponds to a row in the DataFrame.
            # This is the standardized format we want for self.data
            for _, row in jsonl_data.iterrows():
                self.data.append(row.to_dict())
        else:
            raise NotImplementedError(
                "Only JSONL format is supported for CustomDataset."
            )

        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: str | None = None,
        max_loras: int | None = None,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        # load all data if needed
        self.num_available_samples = len(self.data)
        if num_requests <= 0:
            num_requests = self.num_available_samples
            logger.info(
                "num_requests is set to 0 or negative, "
                "so using all available samples: %d",
                num_requests,
            )

        sampled_requests = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["prompt"]

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )

        return sampled_requests


# -----------------------------------------------------------------------------
# Spec Bench Dataset Implementation
# -----------------------------------------------------------------------------


class SpecBench(CustomDataset):
    """
    Implements the SpecBench dataset: https://github.com/hemingkx/Spec-Bench
    Download the dataset using:
    wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
    """  # noqa: E501

    def __init__(self, **kwargs) -> None:
        self.category = kwargs.pop("category", None)
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        self.data = []

        # Load the JSONL file
        jsonl_data = pd.read_json(path_or_buf=self.dataset_path, lines=True)

        # check if the JSONL file has a 'turns' column
        if "turns" not in jsonl_data.columns:
            raise ValueError("JSONL file must contain a 'turns' column.")

        for _, row in jsonl_data.iterrows():
            # sample only from a specific category if specified
            if (not self.category) or (self.category == row["category"]):
                prompt = row["turns"][0]
                self.data.append({"prompt": prompt})

        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(self, **kwargs) -> list:
        # leverage CustomDataset sample
        return super().sample(**kwargs)


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens) for tokens in tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(
            base_msg, add_generation_prompt=True, tokenize=False
        )
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset})."
            )

        # Determine how many poem lines to use.
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = max(round((prefix_len - base_offset) / avg_len), 0)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        ind = 0
        while len(samples) < num_requests:
            extra_lines = random.choices(
                self.data, k=num_input_lines - num_prefix_lines
            )
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            if prompt_len <= input_len:
                samples.append(
                    SampleRequest(
                        prompt=prompt_formatted if return_prompt_formatted else prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        request_id=request_id_prefix + str(ind),
                    )
                )
                ind += 1
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(
        self,
    ):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        self.data = gpt4_df

    def _sample_loaded_data(self, num_requests: int) -> list:
        if num_requests <= len(self.data):
            data = self.data.sample(n=num_requests, random_state=self.random_seed)
        else:
            data = self.data.sample(
                n=num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        return data.values.tolist()

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        max_loras: int | None = None,
        lora_path: str | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        samples = []
        data = self._sample_loaded_data(num_requests=num_requests)
        for i in range(num_requests):
            input_len = int(data[i][2])
            output_len = int(data[i][3])
            lora_req = None
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: set[str] | dict[str, Callable] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        no_stream: bool = False,
        dataset_subset: str | None = None,
        hf_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_stream = not no_stream
        self.hf_name = hf_name or dataset_path
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=self.load_stream,
        )
        if not getattr(self, "disable_shuffle", False):
            self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Instruct Coder Dataset Implementation
# -----------------------------------------------------------------------------


class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    InstructCoder is the dataset designed for general code editing.  It consists
    of 114,239 instruction-input-output triplets, and covers multiple distinct
    code editing scenario.
    """

    DEFAULT_OUTPUT_LEN = 200  # this is the average default output length
    SUPPORTED_DATASET_PATHS = {
        "likaixin/InstructCoder",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = (
                f"{item['input']}\n\n{item['instruction']} Just output "
                "the code, do not include any explanation."
            )

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# MT-Bench Dataset Implementation
# -----------------------------------------------------------------------------


class MTBenchDataset(HuggingFaceDataset):
    """
    MT-Bench Dataset.
    https://huggingface.co/datasets/philschmid/mt-bench

    We create a single turn dataset for MT-Bench.
    This is similar to Spec decoding benchmark setup in vLLM
    https://github.com/vllm-project/vllm/blob/9d98ab5ec/examples/offline_inference/eagle.py#L14-L18
    """  # noqa: E501

    DEFAULT_OUTPUT_LEN = 256  # avg len used in SD bench in vLLM
    SUPPORTED_DATASET_PATHS = {
        "philschmid/mt-bench",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests = []

        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0]

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests
