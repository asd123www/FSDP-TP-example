import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import ModuleList, Linear
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, BackwardPrefetch
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import fully_shard
from transformers import LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM, Gemma2Config, AutoConfig, Gemma2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP, LlamaFlashAttention2, LlamaModel
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os
import gzip
import datetime
import json
import socket
import time
import logging
import functools
from functools import partial
from torch.distributed.tensor import DTensor
import pprint
from torch.distributed.device_mesh import DeviceMesh
import accelerate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor.device_mesh import init_device_mesh


def trace_handler(prof: torch.profiler.profile, dir_name="torch_profile_output",
                  worker_name = None, use_gzip: bool = False,
                  file_prefix="prefilling", device_id=0):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise RuntimeError("Can't create directory: " + dir_name) from e
    if not worker_name:
        worker_name = f"{socket.gethostname()}_{os.getpid()}"
    # Use nanosecond here to avoid naming clash when exporting the trace
    timestamp = time.time_ns()
    file_name = f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json"
    if use_gzip:
        file_name = file_name + ".gz"
    prof.export_chrome_trace(os.path.join(dir_name, file_name))
    # Fix the rank issue for  HolisticTraceAnalysis
    # reference: https://github.com/facebookresearch/HolisticTraceAnalysis/issues/107
    # FIXME: This does not work for json.gz
    # rn_rank = np.random.randint(low=0, high=16, dtype=int) # If there are multiple traces files, then each file should have a unique rank value.
    if use_gzip:
        with gzip.open(os.path.join(dir_name, file_name), mode="rt") as fin:
            data = json.loads(fin.read())
        data["distributedInfo"] = {"rank": device_id} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with gzip.open(os.path.join(dir_name, file_name), 'w') as fout:
            fout.write(json.dumps(data).encode('utf-8')) 
    else:
        with open(os.path.join(dir_name, file_name), "r") as fin:
            data = json.load(fin)
        data["distributedInfo"] = {"rank": device_id} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with open(os.path.join(dir_name, file_name), "w") as fout:
            json.dump(data, fout, indent=2)


def get_param_size(module):
    num_params = sum(p.numel() * p.element_size() for p in module.parameters())
    return num_params


def torch_profile_benchmark(model, optimizer, input_ids, warmup_run, test_run):
    test_run = 5
    rank = dist.get_rank()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    dist.barrier()

    for _ in tqdm(range(warmup_run)) if rank == 0 else range(warmup_run):
        # optimizer.zero_grad()
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()
        # optimizer.step()
        model.zero_grad(set_to_none=True)

    min_iter_time = None
    sum_iter_time = 0
    logging.info("Run torch profiler...")
    outfile_prefix = f"fsdp2_train"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=partial(
            trace_handler, dir_name=torch_profile_dir, use_gzip=True, file_prefix=outfile_prefix, device_id=rank
        )
    ) as prof:
        for idx in range(test_run):
            start_event.record()

            # optimizer.zero_grad()
            outputs = model(input_ids)
            loss = outputs.logits.sum()
            loss.backward()
            # optimizer.step()

            end_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(end_event)
            if min_iter_time is None or iter_time < min_iter_time:
                min_iter_time = iter_time
            sum_iter_time += iter_time
            if rank == 0:
                print(f"{idx + 1}/{test_run}: {iter_time} ms")
            model.zero_grad(set_to_none=True)
    dist.barrier()

    result = {
        "min_iter_time": min_iter_time,
        "avg_iter_time": sum_iter_time / test_run,
    }

    return result


def benchmark(model, optimizer, input_ids, warmup_run, test_run):
    rank = dist.get_rank()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    dist.barrier()

    for _ in tqdm(range(warmup_run)) if rank == 0 else range(warmup_run):
        # optimizer.zero_grad()
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()
        # optimizer.step()
        model.zero_grad(set_to_none=True)

    min_iter_time = None
    sum_iter_time = 0
    for idx in range(test_run):
        start_event.record()

        # optimizer.zero_grad()
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()
        # optimizer.step()

        end_event.record()
        torch.cuda.synchronize()
        iter_time = start_event.elapsed_time(end_event)
        if min_iter_time is None or iter_time < min_iter_time:
            min_iter_time = iter_time
        sum_iter_time += iter_time
        if rank == 0:
            print(f"{idx + 1}/{test_run}: {iter_time} ms")
        model.zero_grad(set_to_none=True)
    dist.barrier()

    result = {
        "min_iter_time": min_iter_time,
        "avg_iter_time": sum_iter_time / test_run,
    }

    return result


def print_size(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    buf = bucket.buffer()
    rank = dist.get_rank()
    if rank == 0:
        print(f"rank {rank}: bucket nelem={buf.numel()}, nbytes={buf.numel() * buf.element_size()}")
    fut = torch.futures.Future()
    fut.set_result(buf)
    return fut


def initialize_model(model_name):
    # Create a Llama model
    hf_token = "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"
    config = AutoConfig.from_pretrained(
        model_name,
        token=hf_token,
    )
    if config.torch_dtype not in {torch.float16, torch.bfloat16}:
        print(f"Change from {config.torch_dtype} to torch.bfloat16", flush=True)
        config.torch_dtype = torch.bfloat16
    if model_name == "meta-llama/Meta-Llama-3.1-405b":
        num_layers = 36
        print(f"Change num_hidden_layers from {config.num_hidden_layers} to {num_layers}", flush=True)
        config.num_hidden_layers = num_layers
    elif model_name == "intlsy/opt-175b-hyperparam":
        num_layers = 46
        print(f"Change num_hidden_layers from {config.num_hidden_layers} to {num_layers}", flush=True)
        config.num_hidden_layers = num_layers
    if dist.get_rank() == 0:
        print("config initialized", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=config.torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if dist.get_rank() == 0:
        print("model initialized on CPU", flush=True)
        print(model)
        print("\n\n\n\n\n")

    return model, config

def set_modules_to_forward_prefetch(layers, num_to_forward_prefetch):
    for i, layer in enumerate(layers):
        if i >= len(layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(layers, num_to_backward_prefetch):
    for i, layer in enumerate(layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

# fully_shard api.
def fsdp_wrap_model(model, model_name, device_mesh):
    if "opt" not in model_name:
        layers = model.model.layers
    else:
        layers = model.model.decoder.layers

    # each transformer layer
    for layer in layers:
        fully_shard(layer, mesh=device_mesh)
    if hasattr(model.model, 'embed_tokens'):
        fully_shard(model.model.embed_tokens, mesh=device_mesh)
    if hasattr(model, 'lm_head'):
        fully_shard(model.lm_head, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)

    # zezhou: apply prefetching.
    set_modules_to_forward_prefetch(layers, num_to_forward_prefetch=2)
    set_modules_to_backward_prefetch(layers, num_to_backward_prefetch=2)

    if dist.get_rank() == 0:
        print("Model wrapped with FSDP-v2 fully_shard")
        print(model)
    return model


def tp_model(model, tp_mesh, tp_attn=False, tp_mlp=True):
    """Apply tensor parallelism to the model using the provided TP mesh."""
    num_layers = len(model.model.layers)
    tp_plan = {}
    for i in range(num_layers):
        p = f"model.layers.{i}."

        if tp_attn:
            attn = model.model.layers[i].self_attn
            # Scale heads per rank
            assert attn.num_heads % tp_mesh.size() == 0
            assert attn.num_key_value_heads % tp_mesh.size() == 0
            attn.num_heads //= tp_mesh.size()
            attn.num_key_value_heads //= tp_mesh.size()
            # num_key_value_groups stays consistent automatically
            attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads

            # Attention
            tp_plan[p + "self_attn.q_proj"] = ColwiseParallel()
            tp_plan[p + "self_attn.k_proj"] = ColwiseParallel()
            tp_plan[p + "self_attn.v_proj"] = ColwiseParallel()
            tp_plan[p + "self_attn.o_proj"] = RowwiseParallel()

        if tp_mlp:
            # MLP
            tp_plan[p + "mlp.gate_proj"] = ColwiseParallel()
            tp_plan[p + "mlp.up_proj"]   = ColwiseParallel()
            tp_plan[p + "mlp.down_proj"] = RowwiseParallel()

    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=tp_plan,
    )
    if dist.get_rank() == 0:
        print(model)

    return model


def run_model(model_name, batch_size, sequence_length, warmup_run, test_run, torch_profile, tp_size):
    device = torch.device("cuda", torch.cuda.current_device())
    model, config = initialize_model(model_name)
    model.train()

    # Create device mesh for parallelization strategies
    world_size = dist.get_world_size()
    if tp_size == 1:
        # FSDP-only mode (original behavior)
        fsdp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        model = fsdp_wrap_model(model, model_name, fsdp_mesh)
    else:
        assert world_size % tp_size == 0
        fsdp_size = world_size // tp_size
        mesh_2d = init_device_mesh("cuda", (fsdp_size, tp_size), mesh_dim_names=("fsdp", "tp"))
        tp_mesh = mesh_2d["tp"]
        fsdp_mesh = mesh_2d["fsdp"]

        model = tp_model(model, tp_mesh, tp_attn=False, tp_mlp=True)
        model = fsdp_wrap_model(model, model_name, fsdp_mesh)

    if rank == 0:
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor)
            print(f"{name}: {param._spec}")

    optimizer = AdamW(model.parameters(), lr=5e-5)
    # Dummy input
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length)).long()
    input_ids = input_ids.cuda()
    if torch_profile == 0:
        result = benchmark(model, optimizer, input_ids, warmup_run, test_run)
    else:
        result = torch_profile_benchmark(model, optimizer, input_ids, warmup_run, test_run)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDP Llama Benchmark')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=2048, help='Sequence length')
    parser.add_argument('--warmup_run', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--test_run', type=int, default=50, help='Test iterations')
    parser.add_argument('--torch_profile', type=int, default=0, help='Run torch profiler')
    parser.add_argument('--tp_size', type=int, default=1, help='TP size for 2D mesh (default: FSDP-only)')
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    assert local_rank == rank % local_world_size
    # set device for each process.
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')

    print(f"rank {rank}: local_rank={local_rank}, local_world_size={local_world_size}", flush=True)
    print(f"rank {rank}: local_rank={local_rank}, local_world_size={local_world_size}, "
          f"world_size={world_size}\n", flush=True, end="")
    dist.barrier()
    torch_profile_dir = "torch_profile_output"

    result = run_model(
        args.model_name, args.batch_size, args.sequence_length,
        args.warmup_run, args.test_run, args.torch_profile, args.tp_size)

    dist.barrier()
    print(f"rank {rank}: {result}\n", end="")

    dist.barrier()
    dist.destroy_process_group()

# pip install transformers==4.47.1
# pip install accelerate
# MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation

# torchrun --nproc_per_node=8 main_fsdp.py --torch_profile 1 --tp_size 1
