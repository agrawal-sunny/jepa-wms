"""DINO-WM joint/latent conditioning, including multiview frame causality."""
import pytest
import torch

from app.vjepa_wm.utils import init_video_model
from app.vjepa_wm.video_wm import VideoWM
from app.plan_common.models.vit import ViTPredictor


@pytest.mark.parametrize('view_fusion', ['token', 'feature'])
@pytest.mark.parametrize('action_dim', [7, 2048])
@pytest.mark.parametrize('use_proprio', [True, False])
@pytest.mark.parametrize('use_sdpa', [True, False])
def test_dino_wm_conditioning(action_dim, use_proprio, use_sdpa, view_fusion):
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(
        device='cpu', enc_type='precomputed', img_size=28, grid_size=2,
        num_views=2, num_frames_pred=3, embed_dim=24, pred_embed_dim=32,
        pred_type='dino_wm', dino_wm_view_fusion=view_fusion, pred_depth=1, pred_num_heads=2,
        action_dim=action_dim, action_emb_dim=16, action_conditioning='feature',
        action_encoder_inpred=False, proprio_dim=7, proprio_emb_dim=8,
        proprio_encoding='feature', use_proprio=use_proprio, use_sdpa=use_sdpa,
    )
    wm = VideoWM(
        encoder, predictor, action_encoder, proprio_encoder,
        enc_type='precomputed', pred_type='dino_wm', img_size=28, grid_size=2,
        num_views=2, tubelet_size_enc=1, action_skip=1, action_dim=action_dim,
        proprio_dim=7, use_proprio=use_proprio, action_conditioning='feature',
        proprio_encoding='feature', action_encoder_inpred=False,
        cfgs_loss=dict(cos_loss_weight=0., l1_loss_weight=0., l2_loss_weight=1., smooth_l1_loss_weight=0.),
    ).eval()
    # Constructing another predictor must not change this one's lazy SDPA mask.
    ViTPredictor(num_patches=1, num_frames=1, dim=8, depth=1, heads=1, mlp_dim=16)
    obs = dict(visual=torch.randn(1, 3, 2, 4, 24), proprio=torch.randn(1, 3, 7))
    actions = torch.randn(1, 3, action_dim, requires_grad=True)
    visual, prop, act = wm.encode(obs, actions)
    predicted, _, predicted_prop = wm.forward_pred(visual, act, prop)
    assert predicted.shape == visual.shape
    loss = wm.compute_loss(predicted, predicted_prop, visual, prop, shift=1)
    assert torch.isfinite(loss['loss'])
    assert ('proprio_l2_loss' in loss) == use_proprio
    loss['loss'].backward()
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert (proprio_encoder is not None) == use_proprio
    # Neither camera in frame 0 may attend to a future frame.
    changed_visual = visual.clone()
    changed_visual[:, 1:] += torch.randn_like(changed_visual[:, 1:]) * 10
    changed, _, _ = wm.forward_pred(changed_visual, act, prop)
    torch.testing.assert_close(predicted[:, :1], changed[:, :1])
    changed_act = wm.encode_act(actions + 1, views=2)
    changed, _, _ = wm.forward_pred(visual, changed_act, prop)
    assert not torch.allclose(predicted, changed)
    obs['proprio'] += 10
    _, changed_prop, _ = wm.encode(obs, actions)
    changed, _, _ = wm.forward_pred(visual, act, changed_prop)
    if use_proprio:
        assert not torch.allclose(predicted, changed)
    else:
        assert prop is predicted_prop is changed_prop is None
        torch.testing.assert_close(predicted, changed)
    # Short contexts used during autoregressive rollout also work.
    short, _, _ = wm.forward_pred(visual[:, :1], act[:, :1], prop[:, :1] if prop is not None else None)
    torch.testing.assert_close(short, predicted[:, :1])


def test_dino_wm_normalize_action_conditioning_bounds_action_features():
    """Unlike AdaLN, dino_wm previously ignored normalize_action_conditioning
    entirely -- unbounded LAM-scale actions could dwarf visual feature RMS
    (see lac_wm/reports/wm_pipeline_handoff.md finding #1)."""
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(
        device='cpu', enc_type='precomputed', img_size=28, grid_size=2,
        num_views=1, num_frames_pred=3, embed_dim=24, pred_embed_dim=32,
        pred_type='dino_wm', pred_depth=1, pred_num_heads=2,
        action_dim=2048, action_emb_dim=16, action_conditioning='feature',
        action_encoder_inpred=False, proprio_dim=7, use_proprio=False,
    )
    wm = VideoWM(
        encoder, predictor, action_encoder, proprio_encoder,
        enc_type='precomputed', pred_type='dino_wm', img_size=28, grid_size=2,
        num_views=1, tubelet_size_enc=1, action_skip=1, action_dim=2048,
        proprio_dim=7, use_proprio=False, action_conditioning='feature',
        action_encoder_inpred=False, normalize_action_conditioning=True,
        cfgs_loss=dict(cos_loss_weight=0., l1_loss_weight=0., l2_loss_weight=1., smooth_l1_loss_weight=0.),
    ).eval()
    # Large-magnitude LAM-scale actions, well outside the visual feature range.
    large_actions = torch.randn(2, 3, 2048) * 50 + 30
    action_features = wm.encode_act(large_actions, views=1)
    per_step = action_features[:, :, 0, :]  # broadcast over patches; check one
    torch.testing.assert_close(per_step.mean(-1), torch.zeros(2, 3), atol=1e-4, rtol=0)
    torch.testing.assert_close(per_step.std(-1, unbiased=False), torch.ones(2, 3), atol=1e-3, rtol=0)
