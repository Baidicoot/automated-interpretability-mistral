from neuron_explainer.explanations.explainer import TokenActivationPairExplainer, ContextSize
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet

from typing import Optional, Any, List, Sequence, Union, Tuple

from llama_cpp import Llama
from llama_autointerp.llama_client import LlamaClient

class LlamaTokenActivationPairExplainer(TokenActivationPairExplainer):
    def __init__(
        self,
        llama: Llama,
        model_name: Optional[str] = None,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        context_size: ContextSize = ContextSize.TWO_K,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.ORIGINAL,
        repeat_non_zero_activations: bool = True,
        cache: bool = False,
        ):
        self.client = LlamaClient(llama, model_name=model_name)

        self.prompt_format = prompt_format
        self.context_size = context_size
        self.cache = cache
        self.few_shot_example_set = few_shot_example_set
        self.repeat_non_zero_activations = repeat_non_zero_activations

        if self.cache == True:
            assert False, "Caching is not yet implemented for LocalTokenActivationPairExplainer"