import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
    weights and biases of a layer to not require gradients.

    Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

    Note: Remove the last linear layer in Alexnet and add your own layer to 
    perform 15 class classification.

    Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################
    alex_net = alexnet(pretrained = True)
    for i in [0,3,6,8,10]:
      alex_net.features[i].bias.requires_grad = False
      alex_net.features[i].weight.requires_grad = False

    self.cnn_layers = nn.Sequential(
      alex_net.features[0],
      alex_net.features[1],
      alex_net.features[2],
      alex_net.features[3],
      alex_net.features[4],
      alex_net.features[5],
      alex_net.features[6],
      alex_net.features[7],
      alex_net.features[8],
      alex_net.features[9],
      alex_net.features[10],
      alex_net.features[11],
      alex_net.features[12],
    )
    for i in [1,4]:
      alex_net.classifier[i].bias.requires_grad = False
      alex_net.classifier[i].weight.requires_grad = False

    self.fc_layers = nn.Sequential(nn.Flatten(),
                                   alex_net.classifier[0],
                                   alex_net.classifier[1],
                                   alex_net.classifier[2],
                                   alex_net.classifier[3],
                                   alex_net.classifier[4],
                                   alex_net.classifier[5],
                                   nn.Linear(4096,15))
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    ############################################################################
    cnn = self.cnn_layers(x)
    model_output = self.fc_layers(cnn)

    ############################################################################
    # Student code end
    ############################################################################

    return model_output
