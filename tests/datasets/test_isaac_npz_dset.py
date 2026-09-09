import numpy as np

from app.plan_common.datasets.isaac_npz_dset import IsaacNPZDataset, load_isaac_npz_slice_train_val


def _episode(path, length=5):
    actions = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
    np.savez_compressed(
        path,
        front_cam=np.zeros((length, 8, 8, 3), dtype=np.uint8),
        wrist_cam=np.ones((length, 8, 8, 3), dtype=np.uint8) * 255,
        action=actions,
        eef_pos=np.zeros((length, 3), dtype=np.float32),
        eef_quat=np.tile([1, 0, 0, 0], (length, 1)).astype(np.float32),
        reward=np.zeros(length, dtype=np.float32),
    )
    return actions


def test_collector_actions_are_shifted_to_jepa_convention(tmp_path):
    actions = _episode(tmp_path / "episode.npz")
    dataset = IsaacNPZDataset([tmp_path], camera_views=["front_cam"])
    obs, aligned_actions, state, reward, info = dataset[0]

    np.testing.assert_allclose(aligned_actions[:-1], actions[1:])
    np.testing.assert_allclose(aligned_actions[-1], 0)
    assert obs["visual"].shape == (5, 3, 8, 8)
    assert obs["proprio"].shape == state.shape == (5, 7)
    assert reward is None
    assert info["camera"] == "front_cam"


def test_both_task_camera_views_share_the_same_schema(tmp_path):
    _episode(tmp_path / "episode.npz")
    dataset = IsaacNPZDataset([tmp_path], camera_views=["front_cam", "wrist_cam"])
    assert dataset.action_dim == dataset.proprio_dim == dataset.state_dim == 7


def test_rejects_unsynchronized_episode(tmp_path):
    _episode(tmp_path / "episode.npz")
    with np.load(tmp_path / "episode.npz") as original:
        values = {key: original[key] for key in original.files}
    values["eef_pos"] = values["eef_pos"][:-1]
    np.savez_compressed(tmp_path / "episode.npz", **values)

    try:
        IsaacNPZDataset([tmp_path], camera_views=["front_cam"])
    except ValueError as error:
        assert "unsynchronized" in str(error)
    else:
        raise AssertionError("unsynchronized trajectory was accepted")


def test_validation_demo_filter_preserves_training_split(tmp_path):
    for demo_type in ("success", "failure", "random"):
        folder = tmp_path / demo_type
        folder.mkdir()
        for index in range(10):
            _episode(folder / f"episode_{index}.npz")
    kwargs = dict(transform=None, data_paths=[tmp_path], split_ratio=0.5)
    _, original = load_isaac_npz_slice_train_val(**kwargs)
    clips, filtered = load_isaac_npz_slice_train_val(**kwargs, val_demo_types=["success", "failure"])

    assert filtered["train"].indices == original["train"].indices
    assert {p.parent.name for p in filtered["train"].filtered_samples} == {"success", "failure", "random"}
    assert {p.parent.name for p in filtered["valid"].filtered_samples} == {"success", "failure"}
    assert set(filtered["valid"].indices) <= set(original["valid"].indices)
    assert not set(filtered["train"].indices) & set(filtered["valid"].indices)
    assert clips["valid"].dataset is filtered["valid"]
    for index in range(len(clips["valid"])):
        assert clips["valid"][index][0]["visual"].shape == (4, 3, 8, 8)


def test_multiview_pairs_front_and_wrist_for_train_and_valid(tmp_path):
    import torch

    for index in range(6):
        _episode(tmp_path / f"episode_{index}.npz")
    clips, trajectories = load_isaac_npz_slice_train_val(
        transform=None, data_paths=[tmp_path], split_ratio=0.5,
        camera_views=["front_cam", "wrist_cam"],
    )
    # More than one camera_view pairs both splits: the model encodes each
    # view separately and concatenates them (VideoWM.encode_obs), so each
    # clip needs a real view axis rather than a randomly-sampled single
    # camera. flatten_camera_batch's old "split into independent samples"
    # workaround is no longer part of this path.
    assert trajectories["train"].dataset.all_camera_views
    assert trajectories["valid"].dataset.all_camera_views
    train_visual = clips["train"][0][0]["visual"]
    assert train_visual.shape == (4, 2, 3, 8, 8)
    assert torch.all(train_visual[:, 0] == 0)  # front_cam
    assert torch.all(train_visual[:, 1] == 1)  # wrist_cam (255 / 255)
    valid_visual = clips["valid"][0][0]["visual"]
    assert valid_visual.shape == (4, 2, 3, 8, 8)
    assert torch.all(valid_visual[:, 0] == 0)
    assert torch.all(valid_visual[:, 1] == 1)


def test_flatten_camera_batch_still_available_as_a_standalone_utility():
    """flatten_camera_batch itself is unchanged; only train.py stopped calling
    it (multiview batches now flow through VideoWM.encode_obs directly)."""
    import torch

    from app.plan_common.datasets.isaac_npz_dset import flatten_camera_batch

    obs = {"visual": torch.arange(2 * 3 * 2 * 1 * 4 * 4).reshape(2, 3, 2, 1, 4, 4).float(), "proprio": torch.zeros(2, 3, 7)}
    action, state, reward = torch.zeros(2, 3, 7), torch.zeros(2, 3, 7), torch.zeros(2, 3)
    flat_obs, flat_action, flat_state, flat_reward = flatten_camera_batch(obs, action, state, reward)
    assert flat_obs["visual"].shape == (4, 3, 1, 4, 4)
    for value in (flat_action, flat_state, flat_reward, flat_obs["proprio"]):
        assert value.shape[0] == 4


def test_latent_action_width_shift_normalization_and_slice(tmp_path):
    import torch
    path = tmp_path / 'episode.npz'
    _episode(path)
    with np.load(path) as episode:
        values = dict(episode)
    latent = np.arange(5 * 2048, dtype=np.float32).reshape(5, 2048)
    latent[0] = 0
    values['latent_action_lam'] = latent
    np.savez(path, **values)
    dataset = IsaacNPZDataset([path], action_key='latent_action_lam', normalize_action=True)
    assert dataset.action_dim == 2048
    assert dataset.proprio_dim == 7
    expected = torch.from_numpy(latent[1:])
    expected = (expected - expected.mean(0)) / expected.std(0)
    torch.testing.assert_close(dataset[0][1][:-1], expected)
    assert torch.count_nonzero(dataset[0][1][-1]) == 0
    sliced = dataset.get_slice(0, 0, 4, 2, 1, 'concat')[1]
    torch.testing.assert_close(sliced, expected.reshape(2, 4096))


def test_latent_action_width_is_read_from_the_cache(tmp_path):
    """action_dim isn't hardcoded to the legacy 4x512 LAM's width -- it's read off
    whatever the cache actually stores, so a narrower LAM (e.g. 32-dim) just works
    once precompute_latent_actions.py has regenerated the cache for it."""
    path = tmp_path / 'episode.npz'
    _episode(path)
    with np.load(path) as episode:
        values = dict(episode)
    values['latent_action_lam'] = np.zeros((5, 32), dtype=np.float32)
    np.savez(path, **values)
    dataset = IsaacNPZDataset([path], action_key='latent_action_lam')
    assert dataset.action_dim == 32


def test_rejects_inconsistent_latent_action_width_across_episodes(tmp_path):
    """The first episode fixes action_dim; a differently-shaped cache elsewhere
    (e.g. an accidentally-pooled 4x512-to-512 array, or a stale mix of LAM
    checkpoints) must still be rejected, not silently reinterpreted."""
    import pytest
    first = tmp_path / 'episode_a.npz'
    second = tmp_path / 'episode_b.npz'
    _episode(first)
    _episode(second)
    with np.load(first) as episode:
        values = dict(episode)
    values['latent_action_lam'] = np.arange(5 * 2048, dtype=np.float32).reshape(5, 2048)
    np.savez(first, **values)
    with np.load(second) as episode:
        values = dict(episode)
    values['latent_action_lam'] = np.zeros((5, 512), dtype=np.float32)
    np.savez(second, **values)
    with pytest.raises(ValueError, match='must have shape'):
        IsaacNPZDataset([first, second], action_key='latent_action_lam')
