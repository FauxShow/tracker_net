import torch
import torch.nn as nn
import argparse
from siamese_dataloader import SiameseDataloader
from networks import TripletNetwork
import numpy as np
import tqdm


def compute_metrics(pos_distances, neg_distances):
    
    min_dist = np.min([np.min(neg_distances), np.min(pos_distances)])
    max_dist = np.max([np.max(neg_distances), np.max(pos_distances)])
    
    thresholds = np.linspace(min_dist, max_dist, 1000)
    results = {}
    num_samples = pos_distances.shape[0]
    for thresh in thresholds:
        tps = np.where(pos_distances <= thresh)[0].shape[0]
        fps = np.where(neg_distances <= thresh)[0].shape[0]
        tpr = tps / num_samples
        fpr = fps / num_samples
        results[thresh] = {'TPR': tpr, 'FPR': fpr}

    TPRat1e_2 = 0
    TPRat1e_3 = 0
    for thresh, result in results.items():
        if result['FPR'] <= 1e-2 and result['TPR'] > TPRat1e_2:
            TPRat1e_2 = result['TPR']
        if result['FPR'] <= 1e-3 and result['TPR'] > TPRat1e_3:
            TPRat1e_3 = result['TPR']

    return TPRat1e_2, TPRat1e_3


def run_benchmark(dataloader, model, device):
    model.eval()

    pdist = nn.PairwiseDistance(p=2)
    pos_distances = np.ndarray((0))
    neg_distances = np.ndarray((0))

    with torch.no_grad():
        for (anchor, positive, negative, labels) in tqdm.tqdm(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            outputs = model(anchor, positive, negative)
            pos_distances = np.append(pos_distances, pdist(outputs[0], outputs[1]).cpu().numpy())
            neg_distances = np.append(neg_distances, pdist(outputs[0], outputs[2]).cpu().numpy())

    TPRat1e_2, TPRat1e_3 = compute_metrics(pos_distances, neg_distances)
    
    print(f"TPR@FPR1e-3: {TPRat1e_3}")
    print(f"TPR@FPR1e-2: {TPRat1e_2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference code for siamese tracker network")
    parser.add_argument('--weights', required=True, type=str, help="path to weights file")
    parser.add_argument('--data', required=True, type=str, help="path to data root")

    args = parser.parse_args()

    device = torch.device("cuda")

    test_dataset = SiameseDataloader(args.data, 'triplet', False, 1e4, 128, 128, device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=6)
    model = TripletNetwork().to(device)
    model.load_state_dict(torch.load(args.weights))
    run_benchmark(test_loader, model, device)
