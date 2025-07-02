import torch
import math

def direction_performance(pred, target):
    p = pred[:, 2:] - pred[:, :2]
    t = target[:, 2:] - target[:, :2]

    cos_sim = torch.sum(p * t, dim=1) / (
        torch.norm(p, dim=1) * torch.norm(t, dim=1) + 1e-6
    )
    angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * (180.0 / math.pi)

    start_err = torch.norm(pred[:, :2] - target[:, :2], dim=1)
    end_err = torch.norm(pred[:, 2:] - target[:, 2:], dim=1)

    return angle.mean().item(), start_err.mean().item(), end_err.mean().item()