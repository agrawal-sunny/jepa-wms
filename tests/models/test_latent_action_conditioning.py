from pathlib import Path

import pytest
import torch
import yaml

from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN


@pytest.mark.parametrize('action_dim', [7, 2048])
def test_384_action_bottleneck_preserves_proprioception(action_dim):
    predictor = vit_predictor_AdaLN(
        img_size=28, patch_size=14, num_frames=2, tubelet_size=1,
        embed_dim=384, predictor_embed_dim=384, depth=1, num_heads=16,
        action_dim=action_dim, action_emb_dim=384, action_encoder_inpred=True,
        proprio_encoding='feature', proprio_emb_dim=16, proprio_tokens=0,
        proprio_encoder_inpred=False, use_proprio=True, use_rope=True,
    ).eval()
    visual = torch.randn(1, 2, 1, 2, 2, 384)
    action = torch.randn(1, 2, action_dim)
    proprio = torch.randn(1, 2, 4, 16, requires_grad=True)
    output, _, output_proprio = predictor(visual, action, proprio)
    assert predictor.action_encoder(action).shape == (1, 2, 384)
    assert predictor.action_conditioning_projection.out_features == 400
    assert output_proprio.shape[-1] == 16
    changed, _, _ = predictor(visual, action, proprio + 1)
    assert not torch.allclose(output, changed)
    (output.square().mean() + output_proprio.square().mean()).backward()
    assert proprio.grad is not None and torch.count_nonzero(proprio.grad)


def test_legacy_default_keeps_400_wide_action_encoder():
    predictor = vit_predictor_AdaLN(
        img_size=28, patch_size=14, num_frames=2, embed_dim=384,
        predictor_embed_dim=384, depth=1, num_heads=16,
        proprio_emb_dim=16, proprio_encoding='feature',
    )
    assert predictor.action_encoder.out_features == 400
    assert isinstance(predictor.action_conditioning_projection, torch.nn.Identity)
    assert not any('action_conditioning_projection' in key for key in predictor.state_dict())


def _drop_subblock(lines, sub_key):
    """Remove a '  <sub_key>:' sub-block (and its indented body) from a block's lines."""
    out = []
    skipping = False
    for line in lines:
        if not skipping and line.strip() == f'{sub_key}:' and line.startswith('  ') and not line.startswith('   '):
            skipping = True
            continue
        if skipping:
            if line and not line.startswith('   '):
                skipping = False
            else:
                continue
        out.append(line)
    return out


def test_latent_config_preserves_comparison_blocks():
    root = Path(__file__).resolve().parents[2] / 'configs/vjepa_wm'
    franka_text = (root / 'isaac_lift_env_franka.yaml').read_text()
    latent_text = (root / 'isaac_lift_env_franka_ur_latent_action.yaml').read_text()
    franka, latent = map(yaml.safe_load, [franka_text, latent_text])
    for key in ['model', 'loss', 'optimization', 'data_aug']:
        franka_value, latent_value = franka[key], latent[key]
        if key == 'model':
            # heads_cfg is a frozen, eval-visualization-only exception: the
            # latent-action run adds a second (UR) decoder alongside the
            # shared "image_head" entry so each embodiment's eval rollout
            # decodes through its own finetuned decoder (see
            # lac_wm/LATENT_ACTION_WM.md and train.py's val_loader_head_keys).
            # It's never trained (optimization.train_heads: false, still
            # compared below), so it doesn't affect the actual architecture
            # comparison this test protects -- only the trained blocks must
            # stay verbatim-identical.
            franka_heads = franka_value.get('heads_cfg', {}).get('architectures', {}).get('image_head')
            latent_heads = latent_value.get('heads_cfg', {}).get('architectures', {}).get('image_head')
            assert franka_heads == latent_heads, 'shared image_head architecture must still match'
            franka_value = {k: v for k, v in franka_value.items() if k != 'heads_cfg'}
            latent_value = {k: v for k, v in latent_value.items() if k != 'heads_cfg'}
        assert franka_value == latent_value
        # Preserve comments/text as well as parsed settings.
        def block(text):
            tail = text.split('\n' + key + ':\n', 1)[1]
            lines = []
            for line in tail.splitlines():
                if line and not line.startswith((' ', '#')):
                    break
                lines.append(line)
            return _drop_subblock(lines, 'heads_cfg') if key == 'model' else lines
        assert block(franka_text) == block(latent_text)
    assert latent['data']['custom']['action_key'] == 'latent_action_lam'
    assert latent['data']['datasets'] == ['IsaacLiftEnvFranka', 'IsaacLiftEnvUR']


@pytest.mark.parametrize('action_dim', [7, 2048])
@pytest.mark.parametrize('use_proprio', [True, False])
def test_factory_proprio_switch_controls_encoding_prediction_and_loss(monkeypatch, action_dim, use_proprio):
    from app.vjepa_wm import utils
    from app.vjepa_wm.video_wm import VideoWM

    class Encoder(torch.nn.Module):
        patch_size = 14

        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, images):
            return torch.ones(len(images), 4, 384)

    monkeypatch.setattr(utils, 'DinoEncoder', Encoder)
    predictor, encoder, action_encoder, proprio_encoder = utils.init_video_model(
        device='cpu', enc_type='dino', img_size=28, num_frames_pred=2,
        pred_type='AdaLN', pred_depth=1, pred_embed_dim=384, embed_dim=384,
        pred_num_heads=16, action_dim=action_dim, action_emb_dim=384,
        action_encoder_inpred=True, proprio_dim=7 if use_proprio else None,
        proprio_emb_dim=16, proprio_tokens=0, proprio_encoder_inpred=False,
        use_proprio=use_proprio,
    )
    assert (proprio_encoder is not None) == use_proprio
    assert predictor.action_encoder.out_features == 384
    assert predictor.predictor_total_embed_dim == (400 if use_proprio else 384)
    wm = VideoWM(encoder, predictor, action_encoder, proprio_encoder, enc_type='dino', pred_type='AdaLN',
                 img_size=28, grid_size=2, tubelet_size_enc=1, action_skip=1,
                 action_dim=action_dim, proprio_dim=7, use_proprio=use_proprio,
                 action_encoder_inpred=True, batchify_video=True,
                 cfgs_loss=dict(cos_loss_weight=0., l1_loss_weight=0., l2_loss_weight=1., smooth_l1_loss_weight=0.)).eval()
    obs = dict(visual=torch.zeros(1, 2, 3, 28, 28), proprio=torch.randn(1, 2, 7))
    actions = torch.randn(1, 2, action_dim)
    video, prop, act = wm.encode(obs, actions)
    predicted, _, predicted_prop = wm.forward_pred(video, act, prop)
    losses = wm.compute_loss(predicted, predicted_prop, video, prop, shift=1)
    assert torch.isfinite(losses['loss'])
    assert ('proprio_l2_loss' in losses) == use_proprio
    obs['proprio'] = obs['proprio'] + 100
    changed_video, changed_prop, changed_act = wm.encode(obs, actions)
    changed, _, _ = wm.forward_pred(changed_video, changed_act, changed_prop)
    if use_proprio:
        assert not torch.allclose(predicted, changed)
    else:
        assert prop is predicted_prop is changed_prop is None
        torch.testing.assert_close(predicted, changed)


def test_normalized_lam_conditioning_bounds_gates_and_gradients():
    torch.manual_seed(234)
    predictor = vit_predictor_AdaLN(
        img_size=28, patch_size=14, num_frames=2, tubelet_size=1,
        embed_dim=96, predictor_embed_dim=96, depth=2, num_heads=4,
        action_dim=2048, action_emb_dim=96, action_encoder_inpred=True,
        proprio_encoding='feature', proprio_emb_dim=0, use_proprio=False,
        use_rope=True, normalize_action_conditioning=True,
    )
    seen = []
    handle = predictor.predictor_blocks[0].register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[1].detach())
    )
    visual = torch.randn(1, 2, 1, 2, 2, 96)
    actions = (torch.randn(1, 2, 2048) * 10000).requires_grad_()
    output, _, _ = predictor(visual, actions)
    output.square().mean().backward()
    handle.remove()
    torch.testing.assert_close(seen[0].square().mean(-1), torch.ones(1, 2))
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
