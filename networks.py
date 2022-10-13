
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(pretrained=False)

        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output



class TripletNetwork(nn.Module):
    """
        Triplet network for image similarity estimation.
    """
    def __init__(self):
        super(TripletNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # initialize the weights
        self.resnet.apply(self.init_weights)

        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        #self.conv3 = nn.Conv2d(128, 16, kernel_size=5, stride=1, padding=2)
        #self.fc1 = nn.Linear(16 * 16 * 16, 512)
        #self.fc2 = nn.Linear(512, 64)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            print("linear layer weight init")
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        '''
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        '''
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, anchor, pos, neg):
        # get three images' features
        anchor_emb = self.forward_once(anchor)
        pos_emb = self.forward_once(pos)
        neg_emb = self.forward_once(neg)
        
        return (anchor_emb, pos_emb, neg_emb)
