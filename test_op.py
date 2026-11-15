from operators.GEMM import GEMM
import tvm
import numpy as np
from RuntimeMod import RuntimeMod
from arch.RTX4090 import RTX4090
from Compute import Compute
from CLayer import CLayer
import os
import time
from datetime import datetime

import argparse

from src.driver.driver import cal_theoretical_occupancy_tile
from src.utils.helper import GPUDetector, byte_size
from typing import Any, Callable, List

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

parser = argparse.ArgumentParser()
parser.add_argument("--shape", nargs="*", type=int, default=[4096, 4096, 4096])
# When using TC, data type should be "float16".
parser.add_argument("--use_tc", dest="use_tc", action="store_true", default=False)
# "float32" for CUDA Core, "float16" for Tensor Core.
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument("--use_pareto", dest="use_pareto", action="store_true", default=False)
parser.add_argument("--retune", action="store_true", default=False,
                    help="ignore the cached config and tune again")
args = parser.parse_args()

M, N, K = args.shape
if args.use_tc:
    assert args.dtype == "float16"
else:
    assert args.dtype == "float32"


def build_wrapper(runtime_mod: RuntimeMod):
    runtime_mod.build()
    return runtime_mod.record


def test_gemm():
    start_time = time.time()
    arch = GPUDetector.create_gpu_instance()

    compute1 = Compute(0, arch, type_size=byte_size(args.dtype))
    rtiles0_all = compute1.solve()
    compute = Compute(1, arch, type_size=byte_size(args.dtype), rtiles0_all=rtiles0_all)
    compute_patterns = compute.solve()
    print(f"Find {len(compute_patterns)} GEMM configs, measuring {len(compute_patterns)}", flush=True)

    compute_patterns = sorted(
        compute_patterns,
        key=lambda x: (x[-1][0] * x[-1][1]) / (x[-1][0] + x[-1][1]),
        reverse=False,
    )
    runtime_mods = []
    for compute_pattern in compute_patterns:
        gemm = GEMM(M, N, K, dtype=args.dtype, name="GEMM")
        runtime_mods.append(RuntimeMod(gemm, arch, compute_pattern))

    res = []
    for runtime_mod in runtime_mods:
        try:
            runtime_mod.build()
            runtime_mod.time_evaluator()
            print(runtime_mod.record, flush=True)
            res.append((runtime_mod.throughput, runtime_mod.compute_pattern))
        except Exception as e:  # noqa: BLE001
            print(
                f"{runtime_mod.compute_pattern} is not a valid config: {str(e)[-200:]}",
                flush=True,
            )
            continue

    end_time = time.time()
    if not res:
        raise RuntimeError(f"no valid GEMM config for {(M, N, K)}")

    res.sort(key=lambda x: x[0], reverse=True)
    runtime = 2 * M * K * N / res[0][0] / 1e9  # unit: ms
    msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{arch.name}]\n"
        f" DeCA GEMM(M={M}, N={N}, K={K}, {args.dtype}), config: {res[0][1]}\n"
        f" throughput: {res[0][0]:.1f} TFLOPS, runtime: {runtime:.4f} ms\n"
        f" total tuning time: {end_time - start_time:.1f} s\n"
    )
    print("✅✅✅ " + msg, flush=True)
    log_path = os.path.join(os.path.dirname(__file__), "deca_performance.log")
    with open(log_path, "a") as f:
        f.write(msg + "\n")


def test_CLayer():
    from src.utils.tune_cache import drop_tune_cache, load_tune_cache

    start_time = time.time()
    arch = GPUDetector.create_gpu_instance()
    # Same GPU and the same operator configuration reuse one cached winner.
    identity = {
        "M": M,
        "N": N,
        "K": K,
        "dtype": args.dtype,
        "use_tc": args.use_tc,
    }
    cached = None if args.retune else load_tune_cache(arch.name, "GEMM", identity, None)

    best_cfg = None
    best_tflops = None
    compile_time = 0.0
    rank_note = "(cached)"
    if cached is not None:
        print(f"Using cached GEMM config: {cached}", flush=True)
        try:
            gemm = GEMM(M, N, K, dtype=args.dtype, name="GEMM")
            runtime_mod = RuntimeMod(gemm, arch, args.use_tc, cached)
            t_build = time.time()
            runtime_mod.build()
            compile_time = time.time() - t_build
            runtime_mod.time_evaluator()
            print(runtime_mod.record, flush=True)
            best_cfg = cached
            best_tflops = runtime_mod.throughput
        except Exception as e:  # noqa: BLE001
            print(f"cached GEMM config failed, retuning: {str(e)[-200:]}", flush=True)
            drop_tune_cache(arch.name, "GEMM", identity)
            best_cfg = None
            start_time = time.time()

    if best_cfg is None:
        _tune_clayer_search(arch, identity, start_time)
        return

    _report_gemm(arch, best_cfg, best_tflops, compile_time, rank_note, start_time)


def _tune_clayer_search(arch, identity, start_time):
    from src.utils.tune_cache import save_tune_cache

    # Register level
    clayer_reg = CLayer(0, arch, use_tc=args.use_tc, shape=args.shape)
    rtiles0 = clayer_reg.solve()

    clayer_shared = CLayer(1, arch, rtiles0=rtiles0, use_tc=args.use_tc, shape=args.shape)
    tile_patterns = clayer_shared.solve()

    # Test the occupancy, waves and com_mem_ratio of all tile patterns
    # for x in tile_patterns:
    #     waves, occupancy = cal_theoretical_occupancy_tile(*x[1], *x[0], dtype=args.dtype)
    #     ratio = (x[-1][0] * x[-1][1] * x[-1][2] * (K // x[-1][2])) / ((x[-1][0] * x[-1][2] + x[-1][1] * x[-1][2] + x[-1][0] * x[-1][1]) * (K // x[-1][2]) * byte_size(args.dtype))
    #     print(f"[WAVES, OCC, COM_MEM_RATIO] {x[1]}, {x[0]}: waves: {waves}, occupancy: {occupancy}, ratio: {ratio}, ", end="")
    #     gemm = GEMM(M, N, K, dtype=args.dtype, name="GEMM")
    #     try:
    #         rt_mod = RuntimeMod(gemm, arch, x)
    #         rt_mod.build()
    #         rt_mod.time_evaluator()
    #         print(rt_mod.throughput, flush=True)
    #     except Exception as _:
    #         print(f"0", flush=True)
    #         continue
    # exit()

    # Sort by the compute memory ratio
    AI = lambda x: (x[-1][0] * x[-1][1] * x[-1][2] * (K // x[-1][2])) / (
        (x[-1][0] * x[-1][2] + x[-1][1] * x[-1][2] + x[-1][0] * x[-1][1])
        * (K // x[-1][2])
        * byte_size(args.dtype)
    )
    sorted_tile_patterns = sorted(tile_patterns, key=AI, reverse=True)

    # Sort by occupancy in descending order
    # AI: Callable[..., float] = lambda x: cal_theoretical_occupancy_tile(*x[1], *x[0])[1]
    # sorted_tile_patterns = sorted(tile_patterns, key=AI, reverse=True)

    # Sort by waves in descending order
    # AI = lambda x: cal_theoretical_occupancy_tile(*x[1], *x[0])[0]
    # sorted_tile_patterns = sorted(tile_patterns, key=AI, reverse=True)

    # Helper functions
    def compute_waves(x):
        return cal_theoretical_occupancy_tile(*x[1], *x[0])[0]
    def compute_occupancy(x):
        return cal_theoretical_occupancy_tile(*x[1], *x[0])[1]
    def compute_com_mem_ratio(x):
        return (x[-1][0] * x[-1][1] * x[-1][2] * (K // x[-1][2])) / (
            (x[-1][0] * x[-1][2] + x[-1][1] * x[-1][2] + x[-1][0] * x[-1][1])
            * (K // x[-1][2])
            * byte_size(args.dtype)
        )

    # Sort by waves in descending order, then by occupancy in descending order
    # sorted_tile_patterns = sorted(tile_patterns, key=lambda x: (compute_waves(x), -compute_occupancy(x)), reverse=False)

    # Sort by waves / compute memory ratio in descending order * occupancy in ascending order
    # sorted_tile_patterns = sorted(tile_patterns, key=lambda x: (compute_waves(x) / (compute_occupancy(x) * compute_com_mem_ratio(x))), reverse=False)

    # Sort by normalized waves / compute memory ratio in descending order * occupancy in ascending order
    def sort_tile_patterns(tile_patterns):
        # Calculate all original values
        computed = []
        for tp in tile_patterns:
            waves = compute_waves(tp)
            occ = compute_occupancy(tp)
            ratio = compute_com_mem_ratio(tp)
            computed.append((tp, waves, occ, ratio))
        # Extract values
        waves_vals = [c[1] for c in computed]
        occ_vals = [c[2] for c in computed]
        ratio_vals = [c[3] for c in computed]
        # Normalize to [0, 1]
        waves_min, waves_max = min(waves_vals), max(waves_vals)
        occ_min, occ_max = min(occ_vals), max(occ_vals)
        ratio_min, ratio_max = min(ratio_vals), max(ratio_vals)
        def norm_waves(v): return (v - waves_min) / (waves_max - waves_min) if waves_max != waves_min else 0.5
        def norm_occ(v): return (v - occ_min) / (occ_max - occ_min) if occ_max != occ_min else 0.5
        def norm_ratio(v): return (v - ratio_min) / (ratio_max - ratio_min) if ratio_max != ratio_min else 0.5
        # Calculate the score after normalization: waves/(occ*ratio), the smaller the better
        sorted_patterns = sorted(
            computed,
            key=lambda x: norm_waves(x[1]) / (norm_occ(x[2]) * norm_ratio(x[3]) + 1e-9),
            reverse=False  # The smaller the better
        )
        return [tp[0] for tp in sorted_patterns]
    if args.use_pareto:
        sorted_tile_patterns = sort_tile_patterns(tile_patterns)

    compute_patterns = sorted_tile_patterns[0:50]
    print(
        f"Find {len(tile_patterns)} GEMM configs, measuring {len(compute_patterns)}",
        flush=True,
    )
    runtime_mods: List[RuntimeMod] = []
    for compute_pattern in compute_patterns:
        gemm = GEMM(M, N, K, dtype=args.dtype, name="GEMM")
        rt_mod = RuntimeMod(gemm, arch, args.use_tc, compute_pattern)
        runtime_mods.append(rt_mod)

    res = []
    compile_time = 0.0
    for runtime_mod in runtime_mods:
        try:
            t_build = time.time()
            runtime_mod.build()
            compile_time += time.time() - t_build
            runtime_mod.time_evaluator()
            print(runtime_mod.record, flush=True)
            res.append(
                (
                    runtime_mod.throughput,
                    runtime_mod.record,
                    runtime_mod.compute_pattern,
                    runtime_mod.get_mod(),
                    runtime_mod.get_sch(),
                )
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"{runtime_mod.compute_pattern} is not a valid config: {str(e)[-200:]}",
                flush=True,
            )
            continue

    if not res:
        raise RuntimeError(f"no valid GEMM config for {(M, N, K)}")

    res.sort(key=lambda x: x[0], reverse=True)
    best_tflops, _, best_cfg, _, _ = res[0]
    rank = compute_patterns.index(best_cfg) + 1
    save_tune_cache(arch.name, "GEMM", identity, best_cfg)
    _report_gemm(
        arch,
        best_cfg,
        best_tflops,
        compile_time,
        f"(rank {rank}/{len(compute_patterns)})",
        start_time,
    )


def _report_gemm(arch, best_cfg, best_tflops, compile_time, rank_note, start_time):
    """Time the torch baseline and print the same summary a search or a cache hit uses."""
    end_time = time.time()
    runtime_ms = 2 * M * K * N / best_tflops / 1e9  # unit: ms
    torch_tflops = torch_matmul_tflops(M, N, K, args.dtype)
    torch_ms = 2 * M * N * K / torch_tflops / 1e9
    msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{arch.name}]\n"
        f" DeCA GEMM(M={M}, N={N}, K={K}, {args.dtype}), config: {best_cfg} "
        f"{rank_note}\n"
        f" throughput: {best_tflops:.1f} TFLOPS, runtime: {runtime_ms:.4f} ms\n"
        f" torch: {torch_tflops:.1f} TFLOPS, runtime: {torch_ms:.4f} ms, "
        f"ratio: {best_tflops / torch_tflops:.3f}\n"
        f" compile time: {compile_time:.1f} s, total tuning time: {end_time - start_time:.1f} s\n"
    )
    print("✅✅✅ " + msg, flush=True)
    log_path = os.path.join(os.path.dirname(__file__), "deca_performance.log")
    with open(log_path, "a") as f:
        f.write(msg + "\n")


def torch_matmul_tflops(M: int, N: int, K: int, dtype: str) -> float:
    """torch.matmul (cuBLAS) baseline; TF32 is disabled so that fp32 runs on CUDA cores."""
    import torch
    from kernels.attn_bench import bench_cuda

    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(M, K, device="cuda", dtype=getattr(torch, dtype))
    b = torch.randn(K, N, device="cuda", dtype=getattr(torch, dtype))
    return 2 * M * N * K / bench_cuda(lambda: a @ b) / 1e12


# test_gemm()
# test_CLayer()

if __name__ == "__main__":
    test_CLayer()
