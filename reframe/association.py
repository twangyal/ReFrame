"""Visible contribution evidence; no geometry optimization or feature training."""
import numpy as np


def extract_evidence(render_with_colors, n_gaussians, mask):
    """Return N×3 foreground/background/total visible pixel mass.

    render_with_colors(c) must render ALL scene Gaussians with frozen geometry,
    unchanged opacity, black background, and N×3 precomputed colors. It returns
    a 3×H×W RGB tensor using the original renderer. No optimizer is involved.
    """
    import torch
    if mask.ndim != 2 or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('mask must be a finite binary H×W tensor')
    if n_gaussians < 1:
        raise ValueError('scene must contain Gaussians')
    # enable_grad is required even when the caller evaluated geometry in no_grad.
    with torch.enable_grad():
        colors = torch.zeros((n_gaussians, 3), dtype=torch.float32, device=mask.device, requires_grad=True)
        rgb = render_with_colors(colors)
        if rgb.shape != (3, *mask.shape) or not torch.isfinite(rgb).all():
            raise ValueError('renderer returned invalid RGB dimensions or values')
        m = mask.to(dtype=rgb.dtype)
        loss = (rgb[0]*m + rgb[1]*(1-m) + rgb[2]).sum()
        e, = torch.autograd.grad(loss, colors)
    if not torch.isfinite(e).all() or (e < -1e-5).any():
        raise ValueError('renderer returned invalid contribution gradients')
    e = e.detach().clamp_min(0)
    if not torch.allclose(e[:, 0]+e[:, 1], e[:, 2], rtol=2e-3, atol=2e-4):
        raise ValueError('contribution conservation failed: foreground + background != total')
    return e


def aggregate_evidence(evidence, *, foreground_threshold=.8, background_threshold=.2,
                       min_visible_frames=2, min_frame_mass=1., max_frame_weight=32.,
                       max_variance=.06):
    """Aggregate T×N×3 evidence. Scores are evidence fractions, not probabilities.

    Each sufficiently visible frame gets min(visible mass, max_frame_weight)
    weight. Variance detects disagreement. Unobserved scores remain NaN and are
    never selected. Caller must exclude failed/missing/low-confidence SAM frames.
    """
    e = np.asarray(evidence, dtype=np.float64)
    params = [foreground_threshold, background_threshold, min_frame_mass, max_frame_weight, max_variance]
    if (not np.isfinite(params).all() or not 0 <= background_threshold < foreground_threshold <= 1
        or min_visible_frames < 1 or int(min_visible_frames) != min_visible_frames
        or min_frame_mass <= 0 or max_frame_weight <= 0 or not 0 <= max_variance <= .25):
        raise ValueError('invalid evidence thresholds')
    if e.ndim != 3 or e.shape[0] == 0 or e.shape[1] == 0 or e.shape[2] != 3:
        raise ValueError('evidence must be nonempty T×N×3')
    if not np.isfinite(e).all() or (e < 0).any():
        raise ValueError('evidence must be finite and nonnegative')
    if not np.allclose(e[..., 0]+e[..., 1], e[..., 2], rtol=2e-3, atol=2e-4):
        raise ValueError('foreground + background must equal total')
    mass = e[..., 2]
    visible = mass >= min_frame_mass
    weights = np.where(visible, np.minimum(mass, max_frame_weight), 0.)
    fraction = np.divide(e[..., 0], mass, out=np.zeros_like(mass), where=mass > 0)
    weight_sum = weights.sum(axis=0)
    score = np.divide((weights*fraction).sum(axis=0), weight_sum,
                      out=np.full(e.shape[1], np.nan), where=weight_sum > 0)
    delta = fraction - np.nan_to_num(score)
    variance = np.divide((weights*delta**2).sum(axis=0), weight_sum,
                         out=np.full(e.shape[1], np.nan), where=weight_sum > 0)
    count = visible.sum(axis=0)
    enough = count >= min_visible_frames
    consistent = variance <= max_variance
    state = np.full(e.shape[1], 'unknown', dtype='<U10')
    state[enough] = 'ambiguous'
    state[enough & consistent & (score >= foreground_threshold)] = 'foreground'
    state[enough & consistent & (score <= background_threshold)] = 'background'
    return {'score': score, 'state': state, 'variance': variance, 'visible_frames': count,
            'foreground_mass': e[..., 0].sum(axis=0), 'background_mass': e[..., 1].sum(axis=0),
            'total_mass': mass.sum(axis=0)}
