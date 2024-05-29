from torch import nn
import torchvision
import torch

class BCNN:
    def __init__(self, device='cuda'):
        self.device = device

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights).to(self.device)

        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Set the manual seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Get the length of class_names (one output unit for each class)
        output_shape = 2

        # Recreate the classifier layer and seed it to the target device
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, 
                            out_features=output_shape, 
                            bias=True)).to(self.device)

        self.model.features = self.model.features[:5]
        # self.model.classifier[1] = torch.nn.Linear(80,output_shape)
        self.auto_transforms = weights.transforms()

    def get_model(self):
        return self.model

    def get_transforms(self):
        return self.auto_transforms
