import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType

# Add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------------------------------------
# Model configuration dataclass
# -----------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Arguments related to model loading and generation parameters.
    """    
    model_name: str
    model_path: str
    world_size: int = 1
    temperature: float = 0.1
    top_p: float = 0.001
    max_tokens: int = 1024
    use_vllm: bool = False
    trust_remote_code: bool = True
    is_evaluate: bool = True # Whether this is an evaluation task

# -----------------------------------------------------------
# FSDP wrapper
# -----------------------------------------------------------
def prepare_fsdp_model(model, world_size):
    """
    Wrap a model with FullyShardedDataParallel (FSDP) for distributed training.
    
    Args:
        model: The PyTorch model to wrap.
        world_size: Number of devices in distributed training.
    
    Returns:
        fsdp_model: The FSDP-wrapped model.
    """
    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    fsdp_model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )

    FSDP.set_state_dict_type(
        fsdp_model, 
        state_dict_type=StateDictType.SHARDED_STATE_DICT, 
        state_dict_config=ShardedStateDictConfig()
    )

    return fsdp_model


# -----------------------------------------------------------
# Model and tokenizer loader
# -----------------------------------------------------------
def get_model_and_tokenizer(config: ModelConfig):
    """
    Load model and tokenizer from a pretrained path.
    
    Args:
        config: ModelConfig containing model parameters.
    
    Returns:
        model: The loaded model (optionally wrapped with FSDP if not evaluation).
        tokenizer: The tokenizer corresponding to the model.
    """
    from transformers import AutoModel, AutoTokenizer
    
    print(f"Loading {config.model_name} Processor/Tokenizer from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        padding_side="left",
        trust_remote_code=config.trust_remote_code,)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading {config.model_name} Model from {config.model_path}...")
    if config.is_evaluate:
        # Evaluation mode: no FSDP wrapping
        model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().cuda()
        model.to(torch.bfloat16)
        return model, tokenizer
    else:
        # Training mode: wrap with FSDP
        model = AutoModel.from_pretrained(
            config.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
        )
        model.to(torch.bfloat16)
        fsdp_model = prepare_fsdp_model(model, config.world_size)
        return fsdp_model, tokenizer