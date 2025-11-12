# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# """vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
#
# import typing
#
# # The environment variables override should be imported before any other
# # modules to ensure that the environment variables are set before any
# # other modules are imported.
# import vllm.env_override  # noqa: F401
#
# MODULE_ATTRS = {
#     "bc_linter_skip": "._bc_linter:bc_linter_skip",
#     "bc_linter_include": "._bc_linter:bc_linter_include",
#     "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
#     "EngineArgs": ".engine.arg_utils:EngineArgs",
#     "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
#     "LLMEngine": ".engine.llm_engine:LLMEngine",
#     "LLM": ".entrypoints.llm:LLM",
#     "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
#     "PromptType": ".inputs:PromptType",
#     "TextPrompt": ".inputs:TextPrompt",
#     "TokensPrompt": ".inputs:TokensPrompt",
#     "ModelRegistry": ".model_executor.models:ModelRegistry",
#     "SamplingParams": ".sampling_params:SamplingParams",
#     "PoolingParams": ".pooling_params:PoolingParams",
#     "ClassificationOutput": ".outputs:ClassificationOutput",
#     "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
#     "CompletionOutput": ".outputs:CompletionOutput",
#     "EmbeddingOutput": ".outputs:EmbeddingOutput",
#     "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
#     "PoolingOutput": ".outputs:PoolingOutput",
#     "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
#     "RequestOutput": ".outputs:RequestOutput",
#     "ScoringOutput": ".outputs:ScoringOutput",
#     "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
# }
#
# def __getattr__(name: str) -> typing.Any:
#     from importlib import import_module
#
#     if name in MODULE_ATTRS:
#         module_name, attr_name = MODULE_ATTRS[name].split(":")
#         module = import_module(module_name, __package__)
#         return getattr(module, attr_name)
#     else:
#         raise AttributeError(f"module {__package__} has no attribute {name}")
#
#
# __all__ = [
#     "__version__",
#     "bc_linter_skip",
#     "bc_linter_include",
#     "__version_tuple__",
#     "LLM",
#     "ModelRegistry",
#     "PromptType",
#     "TextPrompt",
#     "TokensPrompt",
#     "SamplingParams",
#     "RequestOutput",
#     "CompletionOutput",
#     "PoolingOutput",
#     "PoolingRequestOutput",
#     "EmbeddingOutput",
#     "EmbeddingRequestOutput",
#     "ClassificationOutput",
#     "ClassificationRequestOutput",
#     "ScoringOutput",
#     "ScoringRequestOutput",
#     "LLMEngine",
#     "EngineArgs",
#     "AsyncLLMEngine",
#     "AsyncEngineArgs",
#     "initialize_ray_cluster",
#     "PoolingParams",
# ]
