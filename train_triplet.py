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
import matplotlib.pyplot as plt
#from torch.nn import CosineSimilarity

from siamese_dataloader import SiameseDataloader
from networks import TripletNetwork
from triplet_loss import TripletLoss # semihard triplet mining implementation

from benchmark import compute_metrics

def plot_distribution(pos, neg, epoch_num):
    plt.hist(pos, 100, density=True)
    plt.hist(neg, 100, density=True)
    plt.savefig(f"plots/epoch_{epoch_num}.png")
    plt.close()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    triplet_margin_loss = nn.TripletMarginLoss(margin=0.2)
    triplet_semihard_loss = TripletLoss(0.2, device)

    switch_to_semihard = 0 # epoch to switch to semihard triplet mining

    for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
        anchor, positive, negative, = anchor.to(device), positive.to(device), negative.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(anchor, positive, negative)
        if epoch < switch_to_semihard:
            loss = triplet_margin_loss(*outputs)
        else:
            embeddings = torch.concat((outputs[0], outputs[1], outputs[2]))
            labels = labels.swapaxes(0, 1)
            labels = labels.reshape(-1)
            loss = triplet_semihard_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, epoch_num):
    model.eval()
    test_loss = 0
    correct = 0

    print("Running test batch")

    criterion = nn.TripletMarginLoss(margin=0.2)

    pdist = nn.PairwiseDistance(p=2)
    pos_distances = np.ndarray((0))
    neg_distances = np.ndarray((0))

    with torch.no_grad():
        for (anchor, positive, negative, _) in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            outputs = model(anchor, positive, negative)
            test_loss += criterion(*outputs).sum().item()
            pos_distances = np.append(pos_distances, pdist(outputs[0], outputs[1]).cpu().numpy())
            neg_distances = np.append(neg_distances, pdist(outputs[0], outputs[2]).cpu().numpy())

    plot_distribution(pos_distances, neg_distances, epoch_num)

    test_loss /= len(test_loader.dataset)
    TPRat1e_2, TPRat1e_3 = compute_metrics(pos_distances, neg_distances)

    print(f"\nTest:\nAverage Loss: {round(test_loss,5)}")
    print(f"TPR@FPR1e-3: {TPRat1e_3}")
    print(f"TPR@FPR1e-2: {TPRat1e_2}")

    return test_loss, TPRat1e_2, TPRat1e_3

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--data-root', type=str,
                        help='root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma')
    parser.add_argument('--load-ckpt', type=str, 
                        help='Weights to initialise training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
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
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = SiameseDataloader(args.data_root, 'triplet', True, 1e5, 128, 128, device)
    test_dataset = SiameseDataloader(args.data_root, 'triplet', False, 1e4, 128, 128, device)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = TripletNetwork().to(device)
    if args.load_ckpt:
        print(f"Initialising weights from {args.load_ckpt}")
        model.load_state_dict(torch.load(args.load_ckpt))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_model_acc = 0

    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    for epoch in range(0, args.epochs + 0):
        train(args, model, device, train_loader, optimizer, epoch)
        loss, TPRat1e_2, TPRat1e_3 = test(model, device, test_loader, epoch)
        if TPRat1e_2 > best_model_acc:
            best_model_acc = TPRat1e_2
            torch.save(model.state_dict(), f"checkpoints/triplet_network_{round(TPRat1e_2, 5)}_epoch{epoch}.pt")
        scheduler.step()

if __name__ == '__main__':
    main()
