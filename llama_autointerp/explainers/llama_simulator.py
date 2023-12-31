from neuron_explainer.explanations.simulator import (
    ExplanationNeuronSimulator,
    was_token_split,
    compute_predicted_activation_stats_for_token,
)
from neuron_explainer.explanations.explanations import SequenceSimulation, ActivationScale
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import PromptFormat

from llama_autointerp.llama_client import LlamaClient, llama_get_prompt_logits
from llama_cpp import Llama
from llama_cpp.llama_chat_format import ChatFormatter, format_llama2

from typing import Any, Optional, Sequence, List

import numpy.typing as npt
import numpy as np

from itertools import accumulate

def parse_llama_simulation_result(
    logprobs: List[dict[bytes, float]],
    response_tokens: Sequence[bytes],
    text: bytes,
    tokens: Sequence[bytes],
) -> SequenceSimulation:
    # adapted from https://github.com/openai/automated-interpretability/blob/main/neuron-explainer/neuron_explainer/explanations/simulator.py#L188
    # complete with all their TODOs and hacks

    top_logprobs = logprobs

    token_text_offset = list(accumulate((len(t) for t in response_tokens), initial=0))
    scoring_start = text.rfind(b'<start>')

    expected_values = []
    original_sequence_tokens: list[str] = []
    distribution_values: list[list[float]] = []
    distribution_probabilities: list[list[float]] = []

    for i in range(2, len(response_tokens)):
        if len(original_sequence_tokens) == len(tokens):
            # Make sure we haven't hit some sort of off-by-one error.
            # TODO(sbills): Generalize this to handle different tokenizers.
            reached_end = response_tokens[i + 1] == b'<' and response_tokens[i + 2] == b'end'
            assert reached_end, f"{response_tokens[i-3:i+3]}"
            break
        if token_text_offset[i] >= scoring_start:
            # We're looking for the first token after a tab. This token should be the text
            # "unknown" if hide_activations=True or a normalized activation (0-10) otherwise.
            # If it isn't, that means that the tab is not appearing as a delimiter, but rather
            # as a token, in which case we should move on to the next response token.
            if response_tokens[i - 1] == b'\t':
                if response_tokens[i] != b'unknown':
                    #logger.debug("Ignoring tab token that is not followed by an 'unknown' token.")
                    continue

                # j represents the index of the token in a "token<tab>activation" line, barring
                # one of the unusual cases handled below.
                j = i - 2

                current_token = tokens[len(original_sequence_tokens)]
                if current_token + b'\n' == response_tokens[j]:
                    # We're in a case where the tokenization resulted in a newline being folded into
                    # the token. We can't do our usual prediction of activation stats for the token,
                    # since the model did not observe the original token. Instead, we use dummy
                    # values. See the TODO elsewhere in this file about coming up with a better
                    # prompt format that avoids this situation.
                    #logger.debug(
                    #    "Warning: newline before a token<tab>activation line was folded into the token"
                    #)
                    print("Warning: newline before a token<tab>activation line was folded into the token")
                    current_distribution_values = []
                    current_distribution_probabilities = []
                    expected_value = 0.0
                else:
                    # We're in the normal case where the tokenization didn't throw off the
                    # formatting or in the token-was-split case, which we handle the usual way.
                    current_top_logprobs = top_logprobs[i-1]
                    current_top_logprobs = {
                        k.decode("utf-8"): v for k, v in current_top_logprobs.items()
                    }
                    
                    (
                        norm_probabilities_by_distribution_value,
                        expected_value,
                    ) = compute_predicted_activation_stats_for_token(
                        current_top_logprobs,
                    )
                    current_distribution_values = list(
                        norm_probabilities_by_distribution_value.keys()
                    )
                    current_distribution_probabilities = list(
                        norm_probabilities_by_distribution_value.values()
                    )

                original_sequence_tokens.append(current_token)
                distribution_values.append([float(v) for v in current_distribution_values])
                distribution_probabilities.append(current_distribution_probabilities)
                expected_values.append(expected_value)

    return SequenceSimulation(
        tokens=original_sequence_tokens,
        expected_activations=expected_values,
        activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
        distribution_values=distribution_values,
        distribution_probabilities=distribution_probabilities,
    )

def get_top_logprobs(
    logprobs: npt.NDArray[np.single],
    top_k: int,
    model: Llama,
) -> dict[str, float]:
    # get top k logprobs from a logprobs array
    logprobs = logprobs - np.log(np.sum(np.exp(logprobs)))

    top_idxs = np.argpartition(logprobs, -top_k)[-top_k:]
    top_logprobs = {
        model.detokenize([idx]): logprobs[idx] for idx in top_idxs
    }
    return top_logprobs

class LlamaExplanationNeuronSimulator(ExplanationNeuronSimulator):
    def __init__(
        self,
        explanation: str,
        llama: Llama,
        chat_format: ChatFormatter = format_llama2,
        few_shot_example_set: FewShotExampleSet = FewShotExampleSet.ORIGINAL,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
    ):
        self.llama = llama
        self.chat_format = chat_format
        self.explanation = explanation
        self.few_shot_example_set = few_shot_example_set
        self.prompt_format = prompt_format
    
    async def simulate(
        self,
        tokens: Sequence[str],
    ) -> SequenceSimulation:
        prompt = self.make_simulation_prompt(tokens)

        tokens = [t.encode("utf-8") for t in tokens]

        text = self.chat_format(prompt).prompt

        # need to make an async wrapper for this

        logprobs, prompt_tokens = await llama_get_prompt_logits(
            model=self.llama,
            prompt=text,
            logits_processor=None,
        )

        # decode the prompt_tokens into a list of strings
        prompt_tokens = [
            self.llama.detokenize([t]) for t in prompt_tokens
        ]

        top_logprobs = [
            get_top_logprobs(logprob, 15, self.llama) for logprob in logprobs
        ]

        result = parse_llama_simulation_result(
            top_logprobs,
            prompt_tokens,
            text.encode("utf-8"),
            tokens,
        )
        return result