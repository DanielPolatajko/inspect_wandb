# from __future__ import annotations

# import importlib
# import logging
# from functools import wraps
# from typing import Any, Callable

# import weave
# from weave.integrations.patcher import MultiPatcher, NoOpPatcher, SymbolPatcher
# from weave.trace.autopatch import IntegrationSettings, OpSettings
# from weave.trace.op import (
#     _add_accumulator,
# )
# from weave.integrations.openai.openai_sdk import openai_on_input_handler, responses_accumulator, should_use_responses_accumulator, openai_accumulator, openai_on_finish_post_processor, should_use_accumulator, WEAVE_STREAM_START_TIME
# from urllib.parse import urlparse
# from openai import AsyncOpenAI
# from openai.resources.chat import AsyncChat
# from openai.resources.chat.completions import AsyncCompletions
# from typing import override

# _openrouter_patcher: MultiPatcher | None = None

# logger = logging.getLogger(__name__)


# # Surprisingly, the async `client.chat.completions.create` does not pass
# # `inspect.iscoroutinefunction`, so we can't dispatch on it and must write
# # it manually here...
# def create_wrapper_async(settings: OpSettings) -> Callable[[Callable], Callable]:
#     def wrapper(fn: Callable) -> Callable:
#         """We need to do this so we can check if `stream` is used."""

#         def _add_stream_options(fn: Callable) -> Callable:
#             @wraps(fn)
#             async def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
#                 if kwargs.get("stream") and kwargs.get("stream_options") is None:
#                     completion = self
#                     base_url = str(completion._client._base_url)
#                     # Only set stream_options if it targets the OpenAI endpoints
#                     if urlparse(base_url).hostname == "api.openai.com":
#                         kwargs["stream_options"] = {"include_usage": True}

#                 return await fn(self, *args, **kwargs)

#             return _wrapper

#         def _openai_stream_options_is_set(inputs: dict) -> bool:
#             if inputs.get("stream_options") is not None:
#                 return True
#             return False

#         op_kwargs = settings.model_dump()
#         op = weave.op(_add_stream_options(fn), **op_kwargs)
#         op._set_on_input_handler(openai_on_input_handler)
#         return _add_accumulator(
#             op,  # type: ignore
#             make_accumulator=lambda inputs: lambda acc, value: openai_accumulator(
#                 acc,
#                 value,
#                 skip_last=not _openai_stream_options_is_set(inputs),
#                 stream_start_time=inputs.get(WEAVE_STREAM_START_TIME),
#             ),
#             should_accumulate=should_use_accumulator,
#             on_finish_post_processor=openai_on_finish_post_processor,
#         )

#     return wrapper

# def create_wrapper_responses_sync(
#     settings: OpSettings,
# ) -> Callable[[Callable], Callable]:
#     def wrapper(fn: Callable) -> Callable:
#         op_kwargs = settings.model_dump()

#         @wraps(fn)
#         def _inner(*args: Any, **kwargs: Any) -> Any:
#             return fn(*args, **kwargs)

#         op = weave.op(_inner, **op_kwargs)
#         op._set_on_input_handler(openai_on_input_handler)
#         return _add_accumulator(
#             op,  # type: ignore
#             make_accumulator=lambda inputs: lambda acc, value: responses_accumulator(
#                 acc, value
#             ),
#             should_accumulate=should_use_responses_accumulator,
#             on_finish_post_processor=lambda value: value,
#         )

#     return wrapper


# def create_wrapper_responses_async(
# ) -> Callable[[Callable], Callable]:
#     def wrapper(fn: Callable) -> Callable:
#         op_kwargs = {
#             "name": "openrouter.responses.create",
#         }

#         @wraps(fn)
#         async def _inner(*args: Any, **kwargs: Any) -> Any:
#             return await fn(*args, **kwargs)

#         op = weave.op(_inner, **op_kwargs)
#         op._set_on_input_handler(openai_on_input_handler)
#         return _add_accumulator(
#             op,  # type: ignore
#             make_accumulator=lambda inputs: lambda acc, value: responses_accumulator(
#                 acc, value
#             ),
#             should_accumulate=should_use_responses_accumulator,
#             on_finish_post_processor=lambda value: value,
#         )

#     return wrapper

# class PatchedAsyncCompletions(AsyncCompletions):
#     """
#     Patch out the create method with the Weave op which sets trace name dynamically based on the model name
#     """
#     @override
#     def create(*args: Any, **kwargs: Any) -> Any:
#         create_wrapper_async()(super().create)(*args, **kwargs)

# class PatchedAsyncChat(AsyncChat):
#     """
#     Patch out the completions property with the patched AsyncCompletions class
#     """
#     @override
#     def completions(self) -> PatchedAsyncCompletions:
#         return PatchedAsyncCompletions(self._client)

# class PatchedAsyncOpenAI(AsyncOpenAI):
#     """
#     Patch out the completions property with the patched AsyncCompletions class
#     """
#     @override
#     def completions(self) -> PatchedAsyncCompletions:
#         return PatchedAsyncCompletions(self._client)



    
    


# def get_openai_patcher(
#     settings: IntegrationSettings | None = None,
# ) -> MultiPatcher | NoOpPatcher:
#     if settings is None:
#         settings = IntegrationSettings()

#     if not settings.enabled:
#         return NoOpPatcher()

#     global _openrouter_patcher
#     if _openrouter_patcher is not None:
#         return _openrouter_patcher

#     base = settings.op_settings
#     async_completions_create_settings = base.model_copy(
#         update={
#             "name": base.name or "openai.chat.completions.create",
#         }
#     )
#     async_completions_parse_settings = base.model_copy(
#         update={"name": base.name or "openai.chat.completions.parse"}
#     )
#     async_moderation_create_settings = base.model_copy(
#         update={"name": base.name or "openai.moderations.create"}
#     )
#     async_embeddings_create_settings = base.model_copy(
#         update={"name": base.name or "openai.embeddings.create"}
#     )
#     responses_create_settings = base.model_copy(
#         update={"name": base.name or "openai.responses.create"}
#     )
#     async_responses_create_settings = base.model_copy(
#         update={"name": base.name or "openai.responses.create"}
#     )
#     responses_parse_settings = base.model_copy(
#         update={"name": base.name or "openai.responses.parse"}
#     )
#     async_responses_parse_settings = base.model_copy(
#         update={"name": base.name or "openai.responses.parse"}
#     )

#     _openai_patcher = MultiPatcher(
#         [
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.chat.completions"),
#                 "AsyncCompletions.create",
#                 create_wrapper_async(settings=async_completions_create_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.chat.completions"),
#                 "AsyncCompletions.parse",
#                 create_wrapper_async(settings=async_completions_parse_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module(
#                     "openai.resources.beta.chat.completions"
#                 ),
#                 "AsyncCompletions.parse",
#                 create_wrapper_async(settings=async_completions_parse_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.moderations"),
#                 "AsyncModerations.create",
#                 create_wrapper_async(settings=async_moderation_create_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.embeddings"),
#                 "AsyncEmbeddings.create",
#                 create_wrapper_async(settings=async_embeddings_create_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.responses"),
#                 "Responses.create",
#                 create_wrapper_responses_sync(settings=responses_create_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.responses"),
#                 "AsyncResponses.create",
#                 create_wrapper_responses_async(
#                     settings=async_responses_create_settings
#                 ),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.responses"),
#                 "Responses.parse",
#                 create_wrapper_responses_sync(settings=responses_parse_settings),
#             ),
#             SymbolPatcher(
#                 lambda: importlib.import_module("openai.resources.responses"),
#                 "AsyncResponses.parse",
#                 create_wrapper_responses_async(settings=async_responses_parse_settings),
#             ),
#         ]
#     )

#     return _openai_patcher