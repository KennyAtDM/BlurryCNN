from torch import nn
import torchvision
import torch

class BCNN:
    def __init__(self, device='cuda'):
        self.device = device

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights).to(self.device)

        for param in self.model.features.parameters():
            param.requires_grad = True
        
        # Set the manual seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Get the length of class_names (one output unit for each class)
        output_shape = 2

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace = True),
            nn.Linear(in_features=self.model.classifier[1].in_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=output_shape)
        ).to(self.device)

        # Get the transforms used to create our pretrained weights
        self.auto_transforms = weights.transforms()

    def get_model(self):
        return self.model

    def get_transforms(self):
        return self.auto_transforms
