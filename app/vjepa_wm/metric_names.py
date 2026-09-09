"""Readable logging names; internal loss/CSV keys stay independent of W&B."""
import re
from collections import defaultdict


def scalar_metrics(metrics, split, loss_config):
    """Name scalars and summarize reconstruction losses across rollout horizons.

    Reconstruction summaries are means of configured, weighted components over
    the logged horizons. `split/loss` remains the actual optimized objective.
    """
    result = {}
    components = defaultdict(lambda: defaultdict(float))
    for key, value in metrics.items():
        if key.startswith(('train/', 'eval/')):
            namespace, key = key.split('/', 1)
        else:
            namespace = split
        # Periodic trajectory evaluation is collected in the training stats dict.
        if key.startswith('data_traj/'):
            namespace = 'eval'
            key = 'rollout/' + key.removeprefix('data_traj/')
        match = re.fullmatch(r'train_rollout(_parallel)?/(.+)/(\d+)', key)
        if match:
            parallel, metric, step = match.groups()
            component = re.fullmatch(r'(visual|proprio)_(cos|l1|l2|smooth_l1)_loss', metric)
            if component:
                kind, loss_type = component.groups()
                weight = loss_config.get(f'{loss_type}_loss_weight', 0.0)
                if not weight:
                    continue
                group = 'parallel_rollout' if parallel else 'rollout'
                components[(namespace, group, kind)][step] += weight * value
                metric = f'{kind}_{loss_type}_loss'
            group = 'parallel_rollout' if parallel else 'rollout'
            result[f'{namespace}/{group}_step_{step}/{metric}'] = value
            continue
        if key.startswith(('optim/', 'info/')):
            _, key = key.split('/', 1)
            key = key.replace('transition_model/', 'predictor/', 1)
            if key.endswith('/wd'):
                key = key.removesuffix('/wd') + '/weight_decay'
            result[f'optimizer/{key}'] = value
            continue
        if key.startswith('act_'):
            key = 'action_' + key[4:]
        result[f'{namespace}/{key}'] = value
    for (namespace, group, kind), horizons in components.items():
        name = f'{kind}_recon_loss'
        if group == 'parallel_rollout':
            name = 'parallel_' + name
        result[f'{namespace}/{name}'] = sum(horizons.values()) / len(horizons)
    return result
