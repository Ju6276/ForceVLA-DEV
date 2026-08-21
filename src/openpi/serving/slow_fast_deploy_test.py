import json

import numpy as np
import pytest

from openpi.policies import rotation_6d as rot
from openpi.serving import slow_fast_deploy
from openpi.training import fast_dataset

SUMMARY = {
    "pooled_context_shape": [120, 16, 2048],
    "action_chunk_shape": [120, 50, 10],
    "chunk_steps": 5,
    "action_rate_hz": 30.0,
    "context_age_scale_ms": 600.0,
    "slow_rate_range_hz": [5.0, 15.0],
    "slow_latency_range_ms": [50.0, 300.0],
}


def _write_cache(tmp_path, **overrides):
    path = tmp_path / "train.npz"
    path.write_bytes(b"")
    path.with_suffix(".json").write_text(json.dumps({**SUMMARY, **overrides}))
    return path


def test_contract_comes_from_the_artifact_fast_was_trained_on(tmp_path):
    contract = slow_fast_deploy.load_contract(_write_cache(tmp_path))
    assert contract.context_tokens == 16
    assert contract.chunk_steps == 5
    assert contract.action_dims == rot.ROBOT_DIMS
    np.testing.assert_allclose(contract.action_period_s, 1 / 30)
    np.testing.assert_allclose(contract.context_age_scale_s, 0.6)

    config = contract.slow_fast_config(residual_limit=0.02)
    assert config.chunk_steps == 5
    assert config.pose_dims == rot.POSE_DIMS
    np.testing.assert_allclose(config.action_period_s, 1 / 30)


def test_missing_summary_names_the_artifact_instead_of_failing_late(tmp_path):
    path = tmp_path / "train.npz"
    path.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="No Slow-cache summary"):
        slow_fast_deploy.load_contract(path)


def test_contract_rejects_a_cache_the_robot_interface_cannot_consume(tmp_path):
    cache = _write_cache(tmp_path, action_chunk_shape=[120, 50, 7])
    with pytest.raises(ValueError, match="xyz\\+6D\\+gripper"):
        slow_fast_deploy.load_contract(cache)


def test_measured_timing_outside_the_trained_band_is_reported(tmp_path):
    contract = slow_fast_deploy.load_contract(_write_cache(tmp_path))
    assert contract.check_measured_timing(slow_rate_hz=9.0, slow_latency_s=0.12) == []
    problems = contract.check_measured_timing(slow_rate_hz=2.0, slow_latency_s=0.5)
    assert len(problems) == 2
    assert "Slow rate" in problems[0] and "Slow latency" in problems[1]


def test_a_full_rate_train_cache_refuses_to_pose_as_a_trained_band(tmp_path):
    """The recommended train cache is extracted full-rate, so its band is degenerate.

    Reading it as if it were the trained band would call every measured rate a
    violation, which is worse than having no check at all.
    """
    cache = _write_cache(tmp_path, slow_rate_range_hz=[0.0, 0.0], slow_latency_range_ms=None)
    contract = slow_fast_deploy.load_contract(cache)
    assert contract.slow_rate_range_hz is None
    with pytest.raises(ValueError, match="no trained timing band"):
        contract.check_measured_timing(slow_rate_hz=9.0, slow_latency_s=0.12)


def test_the_band_comes_from_the_fast_run_that_redrew_it(tmp_path):
    cache = _write_cache(tmp_path, slow_rate_range_hz=[0.0, 0.0], slow_latency_range_ms=None)
    run = tmp_path / "fast"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"trained_timing": {"slow_rate_range_hz": [4.0, 20.0], "slow_latency_range_ms": [40.0, 250.0]}})
    )

    contract = slow_fast_deploy.load_contract(cache, fast_run=run)

    assert contract.slow_rate_range_hz == (4.0, 20.0)
    np.testing.assert_allclose(contract.slow_latency_range_s, (0.04, 0.25))
    # Shapes still come from the cache; only the band is overridden.
    assert contract.context_tokens == 16 and contract.chunk_steps == 5
    assert contract.check_measured_timing(slow_rate_hz=18.0, slow_latency_s=0.2) == []


def test_a_fast_run_predating_timing_randomization_is_named(tmp_path):
    run = tmp_path / "fast"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ValueError, match="no trained_timing"):
        slow_fast_deploy.load_contract(_write_cache(tmp_path), fast_run=run)


def test_packet_builder_reproduces_the_offline_extraction(tmp_path):
    """The runtime packet must be bit-identical to what extract_slow_cache would store."""
    contract = slow_fast_deploy.load_contract(_write_cache(tmp_path))
    build = slow_fast_deploy.SlowPacketBuilder(contract)
    rng = np.random.default_rng(0)
    # The Teacher pads actions to 32 dimensions and returns the full V-L prefix.
    actions = rng.standard_normal((1, 50, 32)).astype(np.float32)
    context = rng.standard_normal((1, 48, 2048)).astype(np.float32)
    mask = np.ones((1, 48), dtype=bool)
    mask[0, 40:] = False

    chunk, intent, intent_mask = build(actions, context, mask)

    expected_pooled, expected_mask = fast_dataset.pool_context_tokens(context, mask, num_tokens=16)
    np.testing.assert_array_equal(chunk, actions[0, :, : rot.ROBOT_DIMS])
    np.testing.assert_allclose(intent, np.asarray(expected_pooled[0]), atol=1e-6)
    np.testing.assert_array_equal(intent_mask, np.asarray(expected_mask[0]))
    assert chunk.shape == (50, rot.ROBOT_DIMS)
    assert intent.shape == (16, 2048)


def test_packet_builder_rejects_an_unpadded_teacher_action(tmp_path):
    build = slow_fast_deploy.SlowPacketBuilder(slow_fast_deploy.load_contract(_write_cache(tmp_path)))
    with pytest.raises(ValueError, match="7D actions"):
        build(np.zeros((1, 50, 7)), np.zeros((1, 8, 2048)), np.ones((1, 8), dtype=bool))


def test_converted_state_is_what_undoing_delta_actions_expects():
    raw = np.array([0.4, -0.2, 0.3, 3.10, 0.2, -0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    state = slow_fast_deploy.convert_robot_state(raw)
    assert state.shape == (rot.ROBOT_DIMS,)
    # xyz and gripper pass through; the three Euler angles become six continuous values.
    np.testing.assert_allclose(state[:3], raw[:3], atol=1e-6)
    np.testing.assert_allclose(state[rot.POSE_DIMS], raw[6], atol=1e-6)
    np.testing.assert_allclose(rot.sixd_to_rpy(state[3 : rot.POSE_DIMS]), raw[3:6], atol=1e-5)
