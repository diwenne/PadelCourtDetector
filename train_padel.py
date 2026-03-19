"""
Train padel court keypoint detector.

Usage:
    python train_padel.py --exp_id padel_v1 --batch_size 4 --num_epochs 100
"""
from dataset_padel import PadelDataset
import torch
import torch.nn as nn
from base_trainer import train
from base_validator import val
import os
from tensorboardX import SummaryWriter
from tracknet import BallTrackerNet
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--exp_id', type=str, default='padel_v1', help='experiment name')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='epochs between validation')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience (in epochs)')
    parser.add_argument('--steps_per_epoch', type=int, default=500, help='steps per epoch')
    parser.add_argument('--input_height', type=int, default=1088, help='input image height')
    parser.add_argument('--input_width', type=int, default=1920, help='input image width')
    parser.add_argument('--resume', action='store_true', help='resume from last checkpoint')
    args = parser.parse_args()
    
    # Force default LR to 3e-4 if not explicitly passed
    if args.lr == 1e-4:
        args.lr = 3e-4

    print("=== Padel Court Keypoint Detector Training ===")
    print(f"Experiment: {args.exp_id}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Create datasets
    train_dataset = PadelDataset('train', input_height=args.input_height, input_width=args.input_width)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataset = PadelDataset('val', input_height=args.input_height, input_width=args.input_width)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model: 6 output channels (4 keypoints + 2 T anchors)
    model = BallTrackerNet(out_channels=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)

    # Load Pretrained Weights Fine-Tuning
    if not args.resume:
        pretrained_path = './exps/padel_v2/model_best.pt'
        if os.path.exists(pretrained_path):
            print(f"Fine-Tuning: Loading weights from {pretrained_path} (skipping final layer dimension mismatch)")
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model_dict = model.state_dict()
            # Filter out final layer weights where shapes mismatch (5 vs 6 channels)
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            print(f"Successfully loaded {len(filtered_dict)} / {len(model_dict)} tensors.")
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)

    # Setup experiment directory
    exps_path = './exps/{}'.format(args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume and os.path.exists(model_last_path):
        checkpoint = torch.load(model_last_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            val_best_accuracy = checkpoint.get('best_accuracy', 0)
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"Resumed from epoch {start_epoch}, best_accuracy={val_best_accuracy:.4f}, no_improve_count={epochs_without_improvement}")
        else:
            # Old format - just model weights
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights (old format), starting from epoch 0")

    val_best_accuracy = 0 if not args.resume else val_best_accuracy if 'val_best_accuracy' in dir() else 0
    epochs_without_improvement = 0 if not args.resume else epochs_without_improvement if 'epochs_without_improvement' in dir() else 0
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)

        if (epoch > 0) & (epoch % args.val_intervals == 0):
            val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, epoch,
                                                                  output_width=args.input_width//2,
                                                                  output_height=args.input_height//2)
            print(f'Epoch {epoch}: val_loss={val_loss:.4f}, accuracy={accuracy:.4f}')
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/tp', tp, epoch)
            log_writer.add_scalar('Val/fp', fp, epoch)
            log_writer.add_scalar('Val/fn', fn, epoch)
            log_writer.add_scalar('Val/tn', tn, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/accuracy', accuracy, epoch)
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': val_best_accuracy,
                    'epochs_without_improvement': epochs_without_improvement,
                }, model_best_path)
                print(f'  New best model saved! accuracy={accuracy:.4f}')
            else:
                epochs_without_improvement += args.val_intervals
                print(f'  No improvement. Patience: {epochs_without_improvement}/{args.patience}')
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': val_best_accuracy,
                'epochs_without_improvement': epochs_without_improvement,
            }, model_last_path)
            
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {epochs_without_improvement} epochs).")
                break

    print(f"\n=== Training Complete ===")
    print(f"Best accuracy: {val_best_accuracy:.4f}")
    print(f"Model saved to: {model_best_path}")
