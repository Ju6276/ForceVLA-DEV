"""Linear probes on the stale-reference target, fit on train and scored on val.

Two questions motivated the one-off analysis this script replaces. Whether the
staleness head is leaving much on the table relative to what a linear read of its
own inputs can recover, and whether the head should emit the drift alone with the
base-state gap supplied in closed form instead of the base-corrected target.

Neither question is answered by an R2. The drift-only target is the larger of the
two, so it admits a much higher R2 while leaving a larger absolute error, and it is
the absolute error that reaches the arm. Both are reported side by side for that
reason.

A linear probe measures linear readability of these features, nothing more. It is
not an information ceiling: a low value is equally consistent with the quantity
being present but nonlinearly encoded. The `key_state` rows exist to separate the
two possible readings of a weak result, since `S_k` is knowable at deployment but
is deliberately not among the student's inputs.

`--ridge 0` is what the degeneracy check needs. Once `S_k` is in the features the
two targets differ by a linear function of those features, so an unpenalized least
squares must land on the same residual; any penalty, however small, breaks that
identity and leaves two numbers that merely agree to several digits.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from openpi.policies import rotation_6d as rot  # noqa: E402
from openpi.shared import normalize as normalize_lib  # noqa: E402
from openpi.training import fast_dataset  # noqa: E402
from train_fast_residual import _staleness_target  # noqa: E402

# The band the deployed student is trained over. The probe has to see the same
# distribution of context ages, otherwise it is answering a different question.
SLOW_RATE_BAND_HZ = (5.0, 15.0)
SLOW_LATENCY_BAND_S = (0.050, 0.300)


def _pooled_slow_context(cache) -> np.ndarray:
    """Masked mean over the cached Slow context tokens, one vector per row.

    The head reaches this context through a trained projector and attention, so a
    mean is a weaker summary than what the model has. It is still the difference
    between a probe over part of the head's inputs and a probe over all of them.
    """
    tokens = np.asarray(cache.context_tokens, dtype=np.float64)
    mask = np.asarray(cache.context_mask, dtype=np.float64)[..., None]
    pooled = np.sum(tokens * mask, axis=1) / np.maximum(np.sum(mask, axis=1), 1.0)
    return pooled[cache.row_key_positions]


def _design_matrix(arrays, cache, *, include_key_state: bool, include_slow_context: bool) -> np.ndarray:
    """Assemble probe features from what the staleness head sees at inference.

    Force is excluded on purpose: the head is force-blind by construction, so a
    probe that saw force would not be a baseline for it. Everything else the head
    sees is available here, including the cached Slow context, which the tabular
    feature set deliberately leaves out and `include_slow_context` restores. `S_k`
    is a separate case: it is knowable at deployment but is deliberately not an
    input, so it only appears under its own flag.
    """
    key_rows = cache.key_dataset_indices[cache.row_key_positions]
    columns = [
        arrays.state,
        cache.reference_actions[:, 0, :].astype(np.float32),
        cache.time_features,
    ]
    if include_slow_context:
        columns.append(_pooled_slow_context(cache))
    if include_key_state:
        columns.append(arrays.state[key_rows])
    features = np.concatenate([np.asarray(column, dtype=np.float64) for column in columns], axis=-1)
    return np.concatenate([features, np.ones((len(features), 1))], axis=-1)


def _standardize(train_x, *others):
    """Put every column on a comparable scale before a single ridge is applied.

    The pooled Slow context contributes two thousand columns whose scale has nothing
    to do with the ten state columns, so one penalty applied to raw features would
    regularize the two blocks by wildly different amounts. The trailing intercept
    column is left alone.
    """
    mean = np.mean(train_x[:, :-1], axis=0, keepdims=True)
    std = np.std(train_x[:, :-1], axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    def apply(matrix):
        return np.concatenate([(matrix[:, :-1] - mean) / std, matrix[:, -1:]], axis=-1)

    return (apply(train_x), *(apply(other) for other in others))


def _solve(train_x, train_y, ridge: float):
    gram = train_x.T @ train_x
    # Never penalize the intercept: doing so shrinks the target's mean toward zero
    # and shows up as an apparent loss of fit that has nothing to do with features.
    penalty = np.full(train_x.shape[1], ridge)
    penalty[-1] = 0.0
    gram[np.diag_indices_from(gram)] += penalty
    return np.linalg.solve(gram, train_x.T @ train_y)


def _fit_and_score(train_x, train_y, val_x, val_y, *, ridge_grid, dev_fraction: float = 0.2) -> dict:
    """Least squares on train, scored on val, with R2 taken about the val mean.

    The regularizer is chosen on a split held out of train, never on val. With two
    thousand context columns a fixed tiny ridge overfits badly enough to drive the
    cross-set R2 negative, which would look like the context carrying no signal
    when it only means the probe was under-regularized.
    """
    if len(ridge_grid) > 1:
        cut = int(len(train_x) * (1.0 - dev_fraction))
        fit_x, dev_x = train_x[:cut], train_x[cut:]
        fit_y, dev_y = train_y[:cut], train_y[cut:]
        errors = [np.sum(np.square(dev_y - dev_x @ _solve(fit_x, fit_y, ridge))) for ridge in ridge_grid]
        ridge = float(ridge_grid[int(np.argmin(errors))])
    else:
        ridge = float(ridge_grid[0])

    residual = val_y - val_x @ _solve(train_x, train_y, ridge)
    total = val_y - np.mean(val_y, axis=0, keepdims=True)
    return {
        "selected_ridge": ridge,
        "target_rms": float(np.sqrt(np.mean(np.square(val_y)))),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(total))),
        "rotation_target_rms": float(np.sqrt(np.mean(np.square(val_y[:, 3:])))),
        "rotation_residual_rms": float(np.sqrt(np.mean(np.square(residual[:, 3:])))),
    }


def _prepare(target_dir, cache_path, *, chunk_steps, resample_seed=None):
    arrays = fast_dataset.load_stage3_fast_arrays(target_dir, chunk_steps=chunk_steps)
    cache = fast_dataset.load_slow_cache(cache_path)
    if resample_seed is not None:
        # The train cache is extracted full rate so its stored timing is degenerate.
        # Training redraws it every few hundred steps; the probe draws it once, which
        # is why the seed is recorded in the output.
        cache = fast_dataset.resample_slow_cache(
            cache,
            arrays.episode_indices,
            arrays.timestamps,
            slow_rate_range_hz=SLOW_RATE_BAND_HZ,
            ready_delay_range_s=SLOW_LATENCY_BAND_S,
            rng=resample_seed,
        )
    ready = np.arange(len(arrays.dataset_indices)) if cache.row_ready is None else np.flatnonzero(cache.row_ready)
    return arrays, cache, ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-targets", type=pathlib.Path, required=True)
    parser.add_argument("--val-targets", type=pathlib.Path, required=True)
    parser.add_argument("--train-slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--val-slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats-dir", type=pathlib.Path, required=True)
    parser.add_argument("--chunk-steps", type=int, default=5)
    parser.add_argument(
        "--ridge",
        type=float,
        nargs="+",
        default=[1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6],
        help="Ridge grid; the value is selected on a split held out of train. Pass a single value to fix it.",
    )
    parser.add_argument("--timing-seed", type=int, default=0, help="Seed for the train-cache timing redraw.")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    train_arrays, train_cache, train_ready = _prepare(
        args.train_targets, args.train_slow_cache, chunk_steps=args.chunk_steps, resample_seed=args.timing_seed
    )
    val_arrays, val_cache, val_ready = _prepare(args.val_targets, args.val_slow_cache, chunk_steps=args.chunk_steps)

    norm_stats = normalize_lib.load(args.norm_stats_dir)
    pose = rot.POSE_DIMS
    state_to_action_scale = (
        (np.asarray(norm_stats["state"].std, dtype=np.float64)[:pose] + 1e-6)
        / (np.asarray(norm_stats["actions"].std, dtype=np.float64)[:pose] + 1e-6)
    ).astype(np.float32)

    def target(arrays, cache, rows, *, include_base_gap):
        full = _staleness_target(
            arrays,
            cache,
            rows,
            pose_dims=pose,
            state_to_action_scale=state_to_action_scale,
            include_base_gap=include_base_gap,
        )
        return np.asarray(full[:, 0], dtype=np.float64)

    feature_sets = (
        ("tabular", False, False),
        ("tabular_plus_slow_context", False, True),
        ("tabular_plus_key_state", True, False),
        ("all_visible_plus_key_state", True, True),
    )
    results = {}
    for feature_name, include_key_state, include_slow_context in feature_sets:
        kwargs = {"include_key_state": include_key_state, "include_slow_context": include_slow_context}
        train_x, val_x = _standardize(
            _design_matrix(train_arrays, train_cache, **kwargs)[train_ready],
            _design_matrix(val_arrays, val_cache, **kwargs)[val_ready],
        )
        for target_name, include_base_gap in (("base_corrected", True), ("drift_only", False)):
            results[f"{feature_name}/{target_name}"] = _fit_and_score(
                train_x,
                target(train_arrays, train_cache, train_ready, include_base_gap=include_base_gap),
                val_x,
                target(val_arrays, val_cache, val_ready, include_base_gap=include_base_gap),
                ridge_grid=args.ridge,
            )

    report = {
        "what_this_measures": (
            "Linear readability of the stale-reference target from the student's own "
            "inference-time inputs. Not an information ceiling."
        ),
        "features": {
            "tabular": (
                "state, reference action at the current tick, time features. Deliberately "
                "NOT the head's full visible input: the cached Slow context is left out."
            ),
            "tabular_plus_slow_context": (
                "the above plus a masked mean over the cached Slow context tokens, which "
                "covers everything the force-blind staleness head can see"
            ),
            "tabular_plus_key_state": "tabular plus the state of the Slow packet's key row",
            "all_visible_plus_key_state": "slow context and key state together",
        },
        "targets": {
            "base_corrected": "A_null(t) - A_ref(t) + (sigma_state/sigma_action)(S_t - S_key), what the head is trained on",
            "drift_only": "A_null(t) - A_ref(t), the analytic-rebase ablation's target",
        },
        "protocol": {
            "fit_on": "train split",
            "scored_on": "val split",
            "regressor": "ridge least squares on standardized features, unpenalized intercept",
            "ridge_grid": list(args.ridge),
            "ridge_selected_on": "the last 20% of train, never on val" if len(args.ridge) > 1 else "fixed",
            "train_timing_seed": args.timing_seed,
            "slow_rate_band_hz": SLOW_RATE_BAND_HZ,
            "slow_latency_band_s": SLOW_LATENCY_BAND_S,
            "train_rows": int(len(train_ready)),
            "val_rows": int(len(val_ready)),
        },
        "results": results,
    }
    print(json.dumps(report, indent=2))
    header = f"{'features / target':<45}{'R2':>9}{'target RMS':>13}{'residual RMS':>21}{'ridge':>10}"
    print("\n" + header)
    for name, value in results.items():
        print(
            f"{name:<45}{value['r2']:>9.4f}{value['target_rms']:>13.6f}"
            f"{value['residual_rms']:>21.15f}{value['selected_ridge']:>10g}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
