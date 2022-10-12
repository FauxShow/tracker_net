import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from torch.nn import CosineSimilarity

from siamese_dataloader import SiameseDataloader
from networks import TripletNetwork


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.TripletMarginLoss()

    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        outputs = model(anchor, positive, negative)
        loss = criterion(*outputs)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    print("Running test batch")

    criterion = nn.TripletMarginLoss()
    cos = CosineSimilarity()

    with torch.no_grad():
        for (anchor, positive, negative) in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            outputs = model(anchor, positive, negative)
            test_loss += criterion(*outputs).sum().item()
            # calculate fixed-thresh accuracy based on cosine sim between embeddings
            positive_cosine_sims = cos(outputs[0], outputs[1])
            negative_cosine_sims = cos(outputs[0], outputs[2])
            pos_pred = torch.where(positive_cosine_sims > 0.5, 1, 0)  
            neg_pred = torch.where(negative_cosine_sims <= 0.5, 1, 0)  
            correct += pos_pred.sum() + neg_pred.sum()

    test_loss /= len(test_loader.dataset)
    accuracy = correct.item() / (len(test_loader.dataset) * 2) # *2 because we make 2 preds per sample (1pos 1neg)

    print(f"\nTest:\nAverage Loss: {round(test_loss,5)}\nAccuracy: {round(accuracy, 5)}")

    return accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--data-root', type=str,
                        help='root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': False,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = SiameseDataloader(args.data_root, 'triplet', True, 128, 128, device)
    test_dataset = SiameseDataloader(args.data_root, 'triplet', False, 128, 128, device)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = TripletNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    best_model_acc = 0.0

    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        accuracy = test(model, device, test_loader)
        train(args, model, device, train_loader, optimizer, epoch)
        if accuracy > best_model_acc:
            best_model_acc = accuracy
            torch.save(model.state_dict(), f"checkpoints/triplet_network_{round(accuracy, 5)}.pt")
        scheduler.step()

if __name__ == '__main__':
    main()
