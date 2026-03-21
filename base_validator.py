"""
Validation loop for heatmap regression keypoint detection.

Evaluates model predictions against ground-truth keypoints using a distance
threshold (default: 7px at output resolution). Computes TP/FP/FN/TN metrics:

    - TP: Predicted point is within `max_dist` pixels of ground truth
    - FP: Predicted point exists but is >max_dist from GT, or GT is off-screen
    - FN: No prediction but GT point exists on-screen
    - TN: No prediction and no GT point on-screen (both off-screen)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)

Called by train_padel.py every `val_intervals` epochs.
"""
import torch
import numpy as np
import torch.nn.functional as F
from utils import is_point_in_image
from scipy.spatial import distance
from postprocess import postprocess
from tracknet import BallTrackerNet
import argparse
import torch.nn as nn

def val(model, val_loader, criterion, device, epoch, output_width=1280, output_height=720, max_dist=7):
    """Run validation over entire val set and compute keypoint detection metrics.
    
    For each keypoint in each sample, the predicted heatmap is postprocessed
    (threshold + HoughCircles) to extract (x, y). This is compared against the
    ground-truth keypoint coordinate using Euclidean distance.
    
    Args:
        model:         BallTrackerNet instance.
        val_loader:    PyTorch DataLoader for validation set.
        criterion:     Loss function (nn.MSELoss).
        device:        'cuda' or 'cpu'.
        epoch:         Current epoch number (for logging).
        output_width:  Width of the output heatmap (default: 1280, padel uses 960).
        output_height: Height of the output heatmap (default: 720, padel uses 544).
        max_dist:      Maximum pixel distance for a prediction to be considered
                       a true positive (default: 7px).
    
    Returns:
        tuple: (mean_loss, tp, fp, fn, tn, precision, accuracy)
            - mean_loss: Average MSE loss across all validation batches.
            - tp/fp/fn/tn: Cumulative counts across all keypoints and samples.
            - precision: TP / (TP + FP).
            - accuracy: (TP + TN) / (TP + TN + FP + FN).
    
    Note:
        Metrics are cumulative across ALL keypoints (including tom/bottom_t
        anchor channels) and ALL samples in the validation set.
    """
    model.eval()
    losses = []
    tp, fp, fn, tn = 0, 0, 0, 0
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            batch_size = batch[0].shape[0]
            out = model(batch[0].float().to(device))
            kps = batch[2]
            gt_hm = batch[1].float().to(device)
            loss = criterion(F.sigmoid(out), gt_hm)

            pred = F.sigmoid(out).detach().cpu().numpy()
            for bs in range(batch_size):
                num_keypoints = kps.shape[1]
                for kps_num in range(num_keypoints):
                    heatmap = (pred[bs][kps_num] * 255).astype(np.uint8)
                    x_pred, y_pred = postprocess(heatmap, scale=1)
                    x_gt = kps[bs][kps_num][0].item()
                    y_gt = kps[bs][kps_num][1].item()

                    if is_point_in_image(x_pred, y_pred, output_width, output_height) and is_point_in_image(x_gt, y_gt, output_width, output_height):
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < max_dist:
                            tp += 1
                        else:
                            fp += 1
                    elif is_point_in_image(x_pred, y_pred, output_width, output_height) and not is_point_in_image(x_gt, y_gt, output_width, output_height):
                        fp += 1
                    elif not is_point_in_image(x_pred, y_pred, output_width, output_height) and is_point_in_image(x_gt, y_gt, output_width, output_height):
                        fn += 1
                    elif not is_point_in_image(x_pred, y_pred, output_width, output_height) and not is_point_in_image(x_gt, y_gt, output_width, output_height):
                        tn += 1

            eps = 1e-15
            precision = round(tp / (tp + fp + eps), 5)
            accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
            print('val, epoch = {}, iter_id = {}/{}, loss = {}, tp = {}, fp = {}, fn = {}, tn = {}, precision = {}, '
                  'accuracy = {}'.format(epoch, iter_id, len(val_loader), round(loss.item(), 5), tp, fp, fn, tn,
                                         precision, accuracy))
            losses.append(loss.item())
    return np.mean(losses), tp, fp, fn, tn, precision, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to pretrained model')
    args = parser.parse_args()

    val_dataset = courtDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    criterion = nn.MSELoss()

    val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, -1)





