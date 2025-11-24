import os
import sys
import io
import json
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import torch
import ray
from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# -----------------------------------------------------------
# --- Fix: Ensure project roots are in sys.path for imports ---
# -----------------------------------------------------------
verl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../verl/verl"))
print("project_root: ", verl_root)
if verl_root not in sys.path:
    sys.path.insert(0, verl_root)

tools_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if tools_root not in sys.path:
    sys.path.insert(0, tools_root)

from my_tools.model import ModelConfig, get_model_and_tokenizer
from my_tools.dataset import SAT, VSI_Bench
from my_tools.utils import postprocess

from protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from utils import hf_tokenizer
from utils.fs import copy_to_local
from utils.hdfs_io import makedirs
from utils.model import compute_position_id_with_mask
from workers.rollout.hf_rollout import HFRollout
from utils.distributed import initialize_global_process_group

# -----------------------------------------------------------
# --- Environment variables ---
# -----------------------------------------------------------
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8268"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# -----------------------------------------------------------
# --- Helper functions ---
# -----------------------------------------------------------
def save_results(output_path: str, results):
    """
    Save evaluation results to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")

# -----------------------------------------------------------
# --- Main entry point ---
# -----------------------------------------------------------
@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    """
    Run model generation and save outputs to parquet and JSON files.
    """
    hf_config = config.rollout
    hf_config.update({"n": 2, "do_sample": True, "validate": False})
    
    local_rank, rank, world_size = initialize_global_process_group()
    print("world size: ", world_size)
    
    # Load model and tokenizer
    model_config = ModelConfig(
        model_path=config.model.path,
        model_name=config.model.name,
        is_evaluate=config.model.is_evaluate,
        trust_remote_code=config.model.trust_remote_code,
        world_size=world_size
    )
    model, tokenizer = get_model_and_tokenizer(model_config)

    # Initialize HFRollout and start generate
    hf_rollout = HFRollout(model, OmegaConf.create(hf_config), tokenizer)

    # Load datasets
    if config.data.name == 'VSI_Bench':
        dataset = VSI_Bench(config.data.path, config.data.file_name, config.data.question_type)
    elif config.data.name == 'SAT':
        dataset = SAT(config.data.path, config.data.file_name)
    else:
        raise RuntimeError(f"[ERROR] dataset {config.data.name} not found.")

    total_samples = len(dataset.prompts)
    print("total_samples: ", total_samples)

    # Generate model outputs
    model_outputs = hf_rollout.chat_sequences(dataset.prompts)
    final_outputs = postprocess(dataset.prompts, model_outputs, config.data.name)

    # Save results
    df = pd.DataFrame(final_outputs)
    if config.data.name == 'SAT':
        output_parquet_file_path = os.path.join(config.data.output_path, f"outputs_{config.model.name}.parquet")
        output_json_file_path = os.path.join(config.data.output_path, f"outputs_{config.model.name}.json")
    elif config.data.name == 'VSI_Bench':
        output_parquet_file_path = os.path.join(config.data.output_path, f"outputs_{config.model.name}_{config.data.question_type}.parquet")
        output_json_file_path = os.path.join(config.data.output_path, f"outputs_{config.model.name}_{config.data.question_type}.json")

    df.to_parquet(output_parquet_file_path, index=False)
    save_results(output_json_file_path, final_outputs)
    print(f"Finished generation for vsibench.")

if __name__ == "__main__":
    main()