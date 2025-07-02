import torch.nn as nn

def my_loss(pred_class, pred_points, target_points, target_class):
    ce = nn.CrossEntropyLoss()(pred_class, target_class)
    mse = nn.MSELoss()(pred_points, target_points)
    total = 0.4 * ce + 0.6 * mse
    return total, mse, ce