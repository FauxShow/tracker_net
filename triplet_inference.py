import torch
import cv2
import argparse
from torchvision.io import read_image
import torch.nn as nn
import torchvision.transforms as T


from networks import TripletNetwork


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Inference code for triplet tracker network")
    parser.add_argument('--weights', required=True, type=str, help="path to weights file")
    parser.add_argument('--image1', required=True, type=str, help="path to image1")
    parser.add_argument('--image2', required=True, type=str, help="path to image2")

    args = parser.parse_args()

    model = TripletNetwork()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    device = torch.device("cuda")
    model = model.to(device)

    transforms = torch.nn.Sequential(T.Resize((128, 128)))

    image1 = read_image(args.image1).type(torch.float32)
    image2 = read_image(args.image2).type(torch.float32)

    image1 = transforms(image1)
    image2 = transforms(image2)
    
    image1 = image1.to(device).unsqueeze(dim=0)
    image2 = image2.to(device).unsqueeze(dim=0)
   
    dist = nn.PairwiseDistance(p=2)

    with torch.no_grad():
        feature1 = model.forward_once(image1).cpu()
        feature2 = model.forward_once(image2).cpu()
        sim_score = float(dist(feature1, feature2)[0])

    print(f"Similarity score: {round(sim_score, 3)}")

    cv2.imshow('image1', cv2.imread(args.image1))
    cv2.imshow('image2', cv2.imread(args.image2))
    cv2.waitKey(0)



