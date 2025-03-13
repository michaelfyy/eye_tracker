import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetV2(nn.Module):
    def __init__(self, 
                 pupil_output_size: int = 2,  
                 pretrained: bool = True):
        """
        Initializes the EfficientNet-based regression model.
        
        Args:
            pupil_output_size (int): Number of regression outputs for pupil localization.
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super(EfficientNetV2, self).__init__()
        # Load a pretrained EfficientNetV2-M model from torchvision
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.efficientnet_v2_m(weights=weights)
        
        # Replace the classifier with a new one that has the desired number of output features
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.model.classifier[0].p, inplace=True),
            nn.Linear(in_features, pupil_output_size)
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, height, width)
            
        Returns:
            dict: A dictionary with keys 'pupil' containing the respective regression outputs.
        """
        return {'pupil': self.model(x)}

if __name__ == "__main__":
    # Quick test to ensure the model runs correctly.
    model = EfficientNetV2(pupil_output_size=2, pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224)  # Example batch of 4 images
    outputs = model(dummy_input)
    print("Pupil output shape:", outputs['pupil'].shape)  # Expected: (4, 2)