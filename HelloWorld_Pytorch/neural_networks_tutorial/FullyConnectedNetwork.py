import torch

class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, hiddenNodes = 5):
        super(FullyConnectedNetwork, self).__init__()

        # Fully connected layer fron the input pixels to all the hidden nodes
        self.fullyConnectedOne = torch.nn.Sequential(
            torch.nn.Linear(24*24, hiddenNodes),
            torch.nn.Sigmoid()       
            )

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hiddenNodes, 1),
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # Convert the 2 dimensional data (image) into a vector
        out = x.reshape(x.size(0), -1)

        #Apply the layers created at initialization time in order
        out = self.fullyConnectedOne(out)
        out = self.outputLayer(out)

        return out