"""Evaluation metrics (paper Sec. IV-B): IOU, IOU-M, subject accuracy, placement accuracy."""
import torch


def iou(a, b):
    """Intersection-over-union of boxes (..., 4) given as (x1, y1, x2, y2). Differentiable."""
    inter_w = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    inter_h = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(union))


def evaluate(model, batch, chunk=500, hard=False):
    """IOU (all objects), IOU-M (moved objects), subject accuracy, placement accuracy.

    Subject accuracy is the fraction of correctly grounded subjects over an example's steps,
    averaged over examples. Placement accuracy: the predicted box centre of a moved object lies
    inside its gold final box.

    `hard=True` evaluates the way the authors' code does at inference: each step moves only the
    argmax subject, using the argmax reference (see `apply_move`). The model's own flag is restored.
    """
    model.eval()
    preds, weights = [], []
    previous, model.hard = getattr(model, "hard", False), hard
    try:
        with torch.no_grad():
            for start in range(0, len(batch), chunk):
                p, w = model(batch.select(range(start, min(start + chunk, len(batch)))))
                preds.append(p), weights.append(w)
    finally:
        model.hard = previous
    pred, subj_w = torch.cat(preds), torch.cat(weights)
    overlap = iou(pred, batch.final)
    steps = torch.arange(batch.subjects.shape[1])[None] < batch.n_steps[:, None]
    correct = (subj_w.argmax(-1) == batch.subjects) & steps
    cx, cy = (pred[..., 0] + pred[..., 2]) / 2, (pred[..., 1] + pred[..., 3]) / 2
    inside = ((cx >= batch.final[..., 0]) & (cx <= batch.final[..., 2])
              & (cy >= batch.final[..., 1]) & (cy <= batch.final[..., 3]))
    return {"iou": overlap[batch.mask > 0].mean().item(),
            "iou_m": overlap[batch.moved].mean().item(),
            "subject_acc": (correct.sum(-1) / batch.n_steps).mean().item(),
            "placement_acc": inside[batch.moved].float().mean().item()}
