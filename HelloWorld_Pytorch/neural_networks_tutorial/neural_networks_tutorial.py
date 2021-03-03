#Pytorch from scratch...
#Source: https://github.com/ghulten/MLLearning/blob/master/PyTorchTutorials/HelloPyTorch/1-TutorialHelloWorld.py

####
# Load the dada
####
import DataHelpers
from PIL import Image

(xTrainFilePaths, yTrain, xTestFilePaths, yTest) = DataHelpers.LoadRawData('dataset_Eye_Images')

xTrainImages = [Image.open(path) for path in xTrainFilePaths]
xTestImages = [Image.open(path) for path in xTestFilePaths]

###
# Get the data into Pytorch's data structures
###

import torch
from torchvision import transforms

xTrainTensor = torch.stack([transforms.ToTensor()(image) for image in xTrainImages ])
yTrainTensor = torch.Tensor( [ [yValue ] for yValue in yTrain ])

print( len(xTestImages))
print( len(yTest))

xTestTensor = torch.stack( [transforms.ToTensor()(image) for image in xTestImages])

###
#   Move the data onto the GPU (or CPU if there's no cuda device avaliable)
### 

device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
print("Device is:", device)

xTrainTensorDevice = xTrainTensor.to(device)
yTrainTensorDevice = yTrainTensor.to(device)

xTestTensorDevice = xTestTensor.to(device)

###
# Set up for the training run
###

import FullyConnectedNetwork
model = FullyConnectedNetwork.FullyConnectedNetwork(hiddenNodes=5)

# copy model onto device
model = model.to(device)

# Loss function (error metric): Mean Square Error, Binary Cross Error (y, y') := y* -log(y') + (1-y)* -log(1-y') 
# "How hard the algorithm should work to correct any particular mistake that it makes"
lossFunction = torch.nn.BCELoss()

# Back propagation algorithm:
# 1. Forward Propagation
# 2. Calculate Loss (how much error the network makes on each sample)
# 3. Back propagation (how much each node in the network is responsible for the error in the training sample)
# 4. Update/Optimize each weight to reduce the error it is contributing to (f.e. SGD: Stochastic Gradient Descent)
# SGD: change the weights to most decrease the loss; take a step proportional to the learning rate ("lr") in the direction of the gradient

optimizer = torch.optim.SGD(model.parameters(), lr= 1.5e-3)

###
# Train the model on the device
###

import time
startTrainingTime = time.time()

# Just for the helloWorld run, in reality we'd have a convergence criterium here
for i in range(2500):
    # 1. Forward Propagation
    yTrainPredicted = model(xTrainTensorDevice)

    # 2. Calculate the loss
    loss = lossFunction(yTrainPredicted, yTrainTensorDevice)

    # 3. Back propagation
    # reset optimizer
    optimizer.zero_grad()

    # set all the gradients in the network
    loss.backward()

    # 4. Optimize
    # take one step in the direction of the gradient
    optimizer.step()

    #print(" Iteration: %d loss: %.4f"   %   (i, loss.item()))

endTrainingTime = time.time()

print("Training complete.   Time:   %s" %   (endTrainingTime - startTrainingTime))

###
# Test the model
###

# No longer in training mode but in evaluation mode
model.train(mode=False)

yTestPredicted = model(xTestTensor)

predictions = [ 1 if probability > .5 else 0 for probability in yTestPredicted]

correct = [ 1 if predictions[i] == yTest[i] else 0 for i in range( len(predictions) ) ]

print("Accuracy: %.2f" % ( sum(correct) / len(correct) ) )