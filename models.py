import torch.nn as nn
import torch.nn.functional as F

class MushroomClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        """
        Define the layers of the for-now fully connected neural network.
        Input: number of input features and number of output classes
        """
        super(MushroomClassifier, self).__init__()

        # linear hidden layers
        self.fc0 = nn.Linear(in_features=in_features, out_features=100)
        self.fc1 = nn.Linear(100, 50)
        # output layer
        self.fc2 = nn.Linear(50, num_classes)
        # sigmoid function for output
        self.sig = nn.Sigmoid()


    def forward(self, x):
        """
        Define the forward pass of the neural network.
        Returns the tensor after the pass
        """

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.sig(self.fc2(x)) # apply sigmoid for binary classification
        return x

