import pytest

from app.vjepa_wm.metric_names import scalar_metrics


def test_rollout_names_preserve_objective_and_distinguish_horizons():
    named = scalar_metrics({
        'loss': 12.,
        'train_rollout/loss/1': 10.,
        'train_rollout/loss/2': 14.,
        'train_rollout/proprio_l2_loss/1': 2.,
        'train_rollout/proprio_l2_loss/2': 4.,
        'train_rollout/proprio_l1_loss/1': 99.,
        'train_rollout/visual_l2_loss/1': 8.,
        'train_rollout/visual_l2_loss/2': 10.,
        'train_rollout_parallel/proprio_l2_loss/2': 20.,
    }, 'train', {'l2_loss_weight': 1.})
    assert named['train/loss'] == 12.
    assert named['train/rollout_step_2/loss'] == 14.
    assert named['train/proprio_recon_loss'] == 3.
    assert named['train/visual_recon_loss'] == 9.
    assert named['train/parallel_proprio_recon_loss'] == 20.
    assert not any('l1_loss' in key or 'train_rollout' in key for key in named)


def test_namespaces_and_weighted_components():
    named = scalar_metrics({
        'eval/held_out/loss': 5.,
        'data_traj/dataset_actions/loss_average': 7.,
        'train/loss': 6.,
        'act_mean': 1.,
        'info/transition_model/wd': .01,
        'optim/transition_model/grad_norm': 2.,
        'train_rollout/visual_l2_loss/1': 4.,
        'train_rollout/visual_l1_loss/1': 2.,
    }, 'train', {'l2_loss_weight': .5, 'l1_loss_weight': .1})
    assert named['eval/held_out/loss'] == 5.
    assert named['eval/rollout/dataset_actions/loss_average'] == 7.
    assert named['train/loss'] == 6.
    assert named['train/action_mean'] == 1.
    assert named['optimizer/predictor/weight_decay'] == .01
    assert named['optimizer/predictor/grad_norm'] == 2.
    assert named['train/visual_recon_loss'] == pytest.approx(2.2)
