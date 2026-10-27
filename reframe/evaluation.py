"""Source-view diagnostics. These numbers do not validate novel-view geometry."""
import numpy as np


def mask_metrics(rendered_visible_mask, source_mask):
    p=np.asarray(rendered_visible_mask); t=np.asarray(source_mask)
    if p.shape!=t.shape or p.ndim!=2 or not ((p==0)|(p==1)).all() or not ((t==0)|(t==1)).all():
        raise ValueError('masks must be same-size binary H×W arrays')
    p=p.astype(bool);t=t.astype(bool)
    intersection=int((p&t).sum());union=int((p|t).sum())
    fp=int((p&~t).sum());fn=int((~p&t).sum());predicted=int(p.sum())
    return {'iou':intersection/union if union else None,'intersection':intersection,'union':union,
            'false_positive_pixels':fp,'false_negative_pixels':fn,
            'false_positive_fraction':fp/predicted if predicted else None}
