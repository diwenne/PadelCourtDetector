"""
Single-epoch training loop for heatmap regression.

Called by train_padel.py once per epoch. Iterates over the DataLoader up to
`max_iters` batches, computing MSE loss between sigmoid(predicted) and
ground-truth Gaussian heatmaps.

The training loop performs:
    1. Forward pass through the model
    2. Sigmoid activation on raw logits
    3. MSE loss against ground-truth heatmaps
    4. Backpropagation and optimizer step
    5. Prints per-iteration loss for monitoring
"""
import torch.nn.functional as F
import numpy as np

def train(model, train_loader, optimizer, criterion, device, epoch, max_iters=1000):
    """Run one epoch of training.
    
    Args:
        model:        BallTrackerNet instance (moved to device).
        train_loader: PyTorch DataLoader yielding (image, heatmap, keypoints, img_name).
        optimizer:    torch.optim.Adam optimizer.
        criterion:    Loss function (nn.MSELoss).
        device:       'cuda' or 'cpu'.
        epoch:        Current epoch number (for logging).
        max_iters:    Maximum batches per epoch. Caps to len(train_loader) if smaller.
                      Default: 1000. Padel training uses 500.
    
    Returns:
        float: Mean training loss across all iterations this epoch.
    
    Note:
        - Sigmoid is applied to model output before loss computation.
        - Optimizer step uses gradient accumulation (backward → step → zero_grad).
        - Loss is printed per iteration for real-time monitoring.
    """
    model.train()
    losses = []
    max_iters = min(max_iters, len(train_loader))

    for iter_id, batch in enumerate(train_loader):
        out = model(batch[0].float().to(device))
        gt_hm_hp = batch[1].float().to(device)
        loss = criterion(F.sigmoid(out), gt_hm_hp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('train, epoch = {}, iter_id = {}/{}, loss = {}'.format(epoch, iter_id, max_iters, loss.item()))
        losses.append(loss.item())
        if iter_id > max_iters:
            break

    return np.mean(losses)





