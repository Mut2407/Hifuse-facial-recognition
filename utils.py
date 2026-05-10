import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# ========== FOCAL LOSS ==========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ========== WEIGHTED SAMPLER ==========
def make_weighted_sampler(labels):
    from collections import Counter
    counts = Counter(labels)
    num_samples = len(labels)
    weights = [1.0 / counts[l] for l in labels]
    return torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)

# ========== TRAIN ONE EPOCH ==========
@torch.enable_grad()
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler=None, use_amp=True, loss_fn=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    accu_loss, accu_num, sample_num = 0.0, 0.0, 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        sample_num += images.shape[0]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
            pred = model(images)
            loss = loss_fn(pred, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler and isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
            lr_scheduler.step()

        accu_loss += loss.item()
        accu_num += torch.eq(torch.argmax(pred, dim=1), labels).sum().item()

        data_loader.desc = f"[train epoch {epoch}] loss: {accu_loss/(step+1):.4f}, acc: {accu_num/sample_num:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"

    return accu_loss/(step+1), accu_num/sample_num
