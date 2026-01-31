import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B"

# transform model use fp32 for better accuracy
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="float32")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# # Select calibration dataset.
# DATASET_ID = "/dataset/workspace/lim42/datasets/ultra_chat_200k/"
# DATASET_SPLIT = "train_sft"
# NUM_CALIBRATION_SAMPLES = 512
# MAX_SEQUENCE_LENGTH = 2048

# # Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
# ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# def preprocess(example):
#     return {
#         "text": tokenizer.apply_chat_template(
#             example["messages"],
#             tokenize=False,
#         )
#     }

# ds = ds.map(preprocess)

# # Tokenize inputs.
# def tokenize(sample):
#     return tokenizer(
#         sample["text"],
#         padding=False,
#         max_length=MAX_SEQUENCE_LENGTH,
#         truncation=True,
#         add_special_tokens=False,
#     )


# ds = ds.map(tokenize, remove_columns=ds.column_names)

# NOTE: currently only rotations R1, R2, and R4 are available
# R3 and learned R1/R2 rotations will be added in a future release.
# Configure the quantization algorithm to run.
#   * apply spinquant transforms to model to reduce quantization loss
#   * quantize the weights to 4 bit with group size 128
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2"],
        transform_type="random-hadamard",
    ),
    # SpinQuantModifier(
    #     rotations=["R4"],
    #     transform_block_size=256,
    #     transform_type="hadamard",
    # ),
    # GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(
    model=model,
    recipe=recipe,
    pipeline="datafree"
)

# oneshot(
#     model=model,
#     dataset=ds,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#     save_compressed=True,
#     trust_remote_code_model=True,
# )

# # Confirm generations of the quantized model look sane.
# print("\n\n")
# print("========== SAMPLE GENERATION ==============")
# dispatch_for_generation(model)
# input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
#     model.device
# )
# output = model.generate(input_ids, max_new_tokens=100)
# print(tokenizer.decode(output[0]))
# print("==========================================\n\n")

# Save to disk.
SAVE_DIR = MODEL_ID + "-R1R2"
model.save_pretrained(SAVE_DIR, save_compressed=False)
tokenizer.save_pretrained(SAVE_DIR)

# Remove quantization_config from config.json if it exists
import json
import os

config_path = os.path.join(SAVE_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        del config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Removed quantization_config from {config_path}")

