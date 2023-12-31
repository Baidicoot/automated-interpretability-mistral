from neuron_explainer.api_client import ApiClient

from llama_cpp import Llama, LogitsProcessor, llama_get_logits, CreateChatCompletionResponse

from typing import Any, Optional, Union, List

import numpy.typing as npt
import numpy as np

import random

import tqdm

async def llama_get_prompt_logits(
    model: Llama,
    prompt: Union[str, List[int]],
    logits_processor: Optional[LogitsProcessor] = None,
    **kwargs
) -> Any:
    # get the output logits from a prompt

    id = random.randint(0, 1000000)

    model.reset()

    if isinstance(prompt, str):
        prompt = model.tokenize(prompt.encode("utf-8"))

    logits = []

    for token in tqdm.tqdm(prompt, desc=f"evaluating {id}", leave=False):
        model.eval([token])
        token_logits = model._scores[-1, :]

        if logits_processor is not None:
            token_logits = logits_processor(token_logits)
        
        logits.append(token_logits)

    return logits, prompt

def merge_llama_responses(
    responses: List[CreateChatCompletionResponse]
) -> CreateChatCompletionResponse:
    # for some reason, the llama api has a `choices` field that is always a list of length 1
    # this function merges multiple responses of the same prompt into one response
    
    response = {k: v for k, v in responses[0].items() if k != "choices"}
    response["choices"] = []

    for idx, r in enumerate(responses):
        response["choices"].append(r["choices"][0])
        response["choices"][idx]["index"] = idx
    
    return response
    

class LlamaClient(ApiClient):
    def __init__(
        self,
        model: Llama,
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.model_name = model_name
    
    async def make_request(
        self, **kwargs
    ) -> dict[str, Any]:
        n = 1
        if "n" in kwargs:
            n = kwargs["n"]
            del kwargs["n"]
        
        responses = []
        for i in range(n):
            kwargs["model"] = self.model_name
            responses.append(self.model.create_chat_completion(**kwargs))
        
        return merge_llama_responses(responses)