"""Measure what a high-rate control path costs, ours versus the usual one.

Existing fast-slow force VLAs keep the vision-language prefix cached and re-run the
pi0 action expert for the high-rate updates. That path still pays a full flow-matching
denoise per update. Ours replaces it with one deterministic forward of a small student
that only has to model the force-attributable residual. This script times both on the
same device so the ratio is measurable rather than asserted.

The vision-language expert is replaced by a narrow placeholder that is never executed:
we pass None for its tokens and supply the prefix KV cache directly. The action expert's
attention cost is unchanged because the placeholder keeps the real KV geometry, and this
keeps the benchmark runnable on a GPU that cannot hold the 2B backbone.
"""

from __future__ import annotations

import argparse
import time

from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as _gemma
from openpi.models import slow_fast


def _time_ms(fn, *, repeats: int, warmup: int = 10) -> np.ndarray:
    # The two-head student returns a tuple, which has no block_until_ready of its own;
    # calling it on the result only works for a single array, so a timing loop written
    # that way reports the dispatch cost of an unfinished computation at best.
    jax.block_until_ready(fn())
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(samples)


def _param_count(tree) -> int:
    return sum(int(np.prod(v.shape)) for v in jax.tree.leaves(tree))


def benchmark_action_expert(args) -> tuple[float, np.ndarray]:
    """Time one action-expert forward against an already-cached prefix."""
    expert = _gemma.get_config(args.action_expert_variant)
    placeholder = _gemma.Config(
        width=64,
        depth=expert.depth,  # gemma.Module asserts every expert shares a depth
        mlp_dim=128,
        num_heads=expert.num_heads,
        num_kv_heads=expert.num_kv_heads,
        head_dim=expert.head_dim,
    )
    llm = _gemma.Module(configs=[placeholder, expert], embed_dtype="bfloat16")
    # The subclass defines its own init() convenience method, which shadows linen's.
    params = nn.Module.init(llm, jax.random.key(0), method=_gemma.Module.init)

    kv_shape = (expert.depth, 1, args.prefix_tokens, expert.num_kv_heads, expert.head_dim)
    kv_cache = (jnp.zeros(kv_shape, jnp.bfloat16), jnp.zeros(kv_shape, jnp.bfloat16))
    suffix = jnp.zeros((1, args.suffix_tokens, expert.width), jnp.bfloat16)
    positions = jnp.arange(args.prefix_tokens, args.prefix_tokens + args.suffix_tokens)[None]
    mask = jnp.ones((1, args.suffix_tokens, args.prefix_tokens + args.suffix_tokens), bool)

    @jax.jit
    def one_step():
        embedded, _ = llm.apply(params, [None, suffix], positions, mask, kv_cache=kv_cache)
        return embedded[1]

    return _param_count(params), _time_ms(one_step, repeats=args.repeats)


def benchmark_fast_student(args) -> tuple[float, np.ndarray]:
    """Time the one deterministic forward the Fast student needs per command."""
    config = slow_fast.FastResidualConfig(chunk_steps=args.chunk_steps)
    model = slow_fast.FastStudentWithIntentProjector(
        config,
        slow_context_dim=args.slow_context_dim,
        num_intent_tokens=args.intent_tokens,
        rngs=nnx.Rngs(0),
    )
    # Both prediction heads intentionally start at zero for safe training. Leaving
    # them at zero lets XLA prune most of the randomly initialized model, producing
    # a dispatch benchmark rather than the trained model's compute cost.
    model.fast_student.residual_head.kernel.value = jax.random.normal(
        jax.random.key(1), model.fast_student.residual_head.kernel.value.shape
    )
    if model.fast_student.staleness_head is not None:
        model.fast_student.staleness_head.kernel.value = jax.random.normal(
            jax.random.key(2), model.fast_student.staleness_head.kernel.value.shape
        )
    inputs = (
        jnp.zeros((1, args.context_tokens, args.slow_context_dim), jnp.float32),
        jnp.ones((1, args.context_tokens), bool),
        jnp.zeros((1, args.force_window, 6), jnp.float32),
        jnp.ones((1, args.force_window), bool),
        jnp.zeros((1, args.action_dims), jnp.float32),
        jnp.zeros((1, args.action_dims), jnp.float32),
        jnp.zeros((1, 2), jnp.float32),
    )

    @nnx.jit
    def one_step(module=model):
        residual, staleness, _, _ = module(*inputs, train=False)
        return residual, staleness

    return _param_count(nnx.state(model, nnx.Param)), _time_ms(one_step, repeats=args.repeats)


def _report(name: str, params: float, samples: np.ndarray, *, updates: int = 1) -> float:
    median = float(np.median(samples)) * updates
    p95 = float(np.percentile(samples, 95)) * updates
    suffix = f" x {updates} flow steps" if updates > 1 else ""
    print(f"{name}{suffix}")
    print(f"  params  {params / 1e6:8.1f} M")
    print(f"  median  {median:8.2f} ms  ({1000.0 / median:6.1f} Hz)")
    print(f"  p95     {p95:8.2f} ms  ({1000.0 / p95:6.1f} Hz)")
    return median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--flow-steps", type=int, default=10, help="denoise steps the action expert pays per update")
    parser.add_argument("--prefix-tokens", type=int, default=816, help="cached vision-language prefix length")
    parser.add_argument("--suffix-tokens", type=int, default=51, help="action chunk tokens plus the state token")
    parser.add_argument("--action-expert-variant", default="gemma_300m")
    parser.add_argument("--chunk-steps", type=int, default=5)
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--intent-tokens", type=int, default=2)
    parser.add_argument("--slow-context-dim", type=int, default=2048)
    parser.add_argument("--force-window", type=int, default=10)
    parser.add_argument("--action-dims", type=int, default=10)
    parser.add_argument("--budget-hz", type=float, default=100.0)
    args = parser.parse_args()

    print(f"device: {jax.devices()[0]}\n")
    expert_params, expert_samples = benchmark_action_expert(args)
    expert_ms = _report("cached-prefix action expert", expert_params, expert_samples, updates=args.flow_steps)
    print()
    student_params, student_samples = benchmark_fast_student(args)
    student_ms = _report("Fast residual student", student_params, student_samples)

    budget_ms = 1000.0 / args.budget_hz
    print(f"\nspeedup {expert_ms / student_ms:.1f}x at a {args.budget_hz:.0f} Hz budget of {budget_ms:.1f} ms")
    for name, value in (("action expert", expert_ms), ("Fast student", student_ms)):
        print(f"  {name:16s} {'fits' if value < budget_ms else 'does not fit'} ({value:.1f} ms)")


if __name__ == "__main__":
    main()
