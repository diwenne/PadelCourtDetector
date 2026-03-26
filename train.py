"""
Unified training script for court keypoint detection.
Supports multiple sports (padel, pickleball).

Usage:
    python train.py --sport padel --exp_id padel_v5
    python train.py --sport pickleball --exp_id pickleball_v1
"""
import torch
import torch.nn as nn
import os
import argparse
from tensorboardX import SummaryWriter
from tracknet import BallTrackerNet
from base_trainer import train
from base_validator import val

def get_dataloader(sport, mode, batch_size, input_height, input_width):
    if sport == 'padel':
        from padel.dataset import PadelDataset
        dataset = PadelDataset(mode, input_height=input_height, input_width=input_width)
    elif sport == 'pickleball':
        from pickleball.dataset import PickleballDataset
        dataset = PickleballDataset(mode, input_height=input_height, input_width=input_width)
    else:
        raise ValueError(f"Unknown sport: {sport}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, default='padel', choices=['padel', 'pickleball'], help='sport type')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--exp_id', type=str, required=True, help='experiment name (e.g. pickleball_v1)')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='epochs between validation')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--steps_per_epoch', type=int, default=500, help='steps per epoch')
    parser.add_argument('--input_height', type=int, default=1088, help='input image height')
    parser.add_argument('--input_width', type=int, default=1920, help='input image width')
    parser.add_argument('--resume', action='store_true', help='resume from last checkpoint')
    parser.add_argument('--model_path', type=str, default='', help='pretrained weights for fine-tuning')
    args = parser.parse_args()

    print(f"=== Unified Court Detector Training: {args.sport.upper()} ===")
    print(f"Experiment: {args.exp_id}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Data loaders
    train_loader = get_dataloader(args.sport, 'train', args.batch_size, args.input_height, args.input_width)
    val_loader = get_dataloader(args.sport, 'val', args.batch_size, args.input_height, args.input_width)

    # Model: 6 output channels (4 keypoints + 2 midpoints)
    model = BallTrackerNet(out_channels=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)

    # Pretrained / Fine-tuning
    if not args.resume and args.model_path:
        if os.path.exists(args.model_path):
            print(f"Fine-Tuning: Loading weights from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model_dict = model.state_dict()
            # Filter for same shapes
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            print(f"Successfully loaded {len(filtered_dict)} / {len(model_dict)} tensors.")
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)

    # Logging
    exps_path = os.path.join('exps', args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    os.makedirs(tb_path, exist_ok=True)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    start_epoch = 0
    val_best_accuracy = 0
    epochs_without_improvement = 0

    if args.resume and os.path.exists(model_last_path):
        checkpoint = torch.load(model_last_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        val_best_accuracy = checkpoint.get('best_accuracy', 0)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/loss', train_loss, epoch)

        if epoch % args.val_intervals == 0:
            val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, epoch, 
                                                                output_width=args.input_width//2, 
                                                                output_height=args.input_height//2)
            print(f'Epoch {epoch}: val_loss={val_loss:.4f}, accuracy={accuracy:.4f}')
            log_writer.add_scalar('Val/accuracy', accuracy, epoch)
            
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': val_best_accuracy,
                }, model_best_path)
                print(f"  New best model saved!")
            else:
                epochs_without_improvement += args.val_intervals
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': val_best_accuracy,
                'epochs_without_improvement': epochs_without_improvement,
            }, model_last_path)

    print(f"Training Complete. Best Accuracy: {val_best_accuracy:.4f}")
