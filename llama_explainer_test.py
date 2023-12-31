import os
import tqdm

os.environ["OPENAI_API_KEY"] = "put-key-here"

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score

from llama_cpp import Llama
from llama_autointerp.explainers.llama_explainer import LlamaTokenActivationPairExplainer
from llama_autointerp.explainers.llama_simulator import LlamaExplanationNeuronSimulator

import asyncio

explainer_model = Llama(
    model_path="models/llama-2-7b-chat.Q6_K.gguf",
    chat_format="llama-2",
    n_ctx=2049,
    n_gpu_layers=99,
    n_threads=7,
)

# Load a neuron record.
neuron_record = load_neuron(9, 6236)

# Grab the activation records we'll need.
slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
train_activation_records = neuron_record.train_activation_records(
    activation_record_slice_params=slice_params
)
valid_activation_records = neuron_record.valid_activation_records(
    activation_record_slice_params=slice_params
)

# Generate an explanation for the neuron.
explainer = LlamaTokenActivationPairExplainer(
    llama=explainer_model,
    model_name="llama-2",
    prompt_format=PromptFormat.HARMONY_V4,
)

explanation = ' words related to family law, specifically same-sex marriage and adoption rights.'

for i in range(10):
    #explanations = asyncio.run(explainer.generate_explanations(
    #    all_activation_records=train_activation_records,
    #    max_activation=calculate_max_activation(train_activation_records),
    #    num_samples=1,
    #))
    #assert len(explanations) == 1
    #explanation = explanations[0]
    print(f"{explanation=}")

    simulator = UncalibratedNeuronSimulator(
        LlamaExplanationNeuronSimulator(
            explanation=explanation,
            llama=explainer_model,
        )
    )
    scored_simulation = asyncio.run(simulate_and_score(simulator, valid_activation_records))
    print(f"score={scored_simulation.get_preferred_score():.2f}")