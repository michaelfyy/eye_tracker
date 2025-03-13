import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiviewNet(nn.Module):
    def __init__(self, 
                 pupil_output_size: int = 3, 
                 gaze_output_size: int = 3,
                 num_channels: int = 4, 
                 pretrained: bool = True):
        """
        Initializes the EfficientNet-based regression model.
        
        Args:
            pupil_output_size (int): Number of regression outputs for pupil localization.
            gaze_output_size (int): Number of regression outputs for gaze regression.
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super(MultiviewNet, self).__init__()
        # Load a pretrained EfficientNetV2-M model from torchvision
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        self.base = models.efficientnet_v2_m(weights=weights)
        
        # Obtain the number of features from the current classifier layer.
        # EfficientNetV2-M classifier is typically a Sequential with a Dropout and a Linear layer.
        in_features = self.base.classifier[1].in_features
        concat_features = in_features * num_channels
        
        # Remove the original classifier by replacing it with an identity module.
        self.base.classifier = nn.Identity()
        
        # Define a regression head for pupil localization.
        self.pupil_regressor = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pupil_output_size)
        )
        
        # Define a regression head for gaze vector regression.
        self.gaze_regressor = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, gaze_output_size)
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch, views, channels, height, width)
            
        Returns:
            dict: A dictionary with keys 'pupil' and 'gaze' containing the respective regression outputs.
        """
        batch, views, channels, height, width = x.shape

        # Reshape to combine batch and view dimensions.
        x = x.view(batch * views, channels, height, width)  # (batch*views, channels, height, width)
        
        # Pass all views through the feature extractor.
        features = self.base(x)  # shape: (batch*views, in_features)
        
        # Reshape to (batch, views, in_features).
        features = features.view(batch, views, -1)
        
        # Concatenate features from all views along the feature dimension.
        features_cat = features.view(batch, -1)  # shape: (batch, views * in_features)
        
        # Pass concatenated features through both regression heads.
        pupil_out = self.pupil_regressor(features_cat)
        gaze_out_raw = self.gaze_regressor(features_cat)
        gaze_out_normalized = F.normalize(gaze_out_raw)
        
        return {'pupil': pupil_out, 'gaze': gaze_out_normalized}

if __name__ == "__main__":
    # Quick test to ensure the model runs correctly.
    # Let's assume we have 4 views per sample and each image is 3x224x224.
    num_views = 4
    batch_size = 32
    model = MultiviewNet(pupil_output_size=3, gaze_output_size=3, num_channels=num_views, pretrained=True)
    # Create a dummy input of shape (batch, views, channels, height, width).
    dummy_input = torch.randn(batch_size, num_views, 3, 224, 224)  # Batch of samples
    outputs = model(dummy_input)
    print("Pupil output shape:", outputs['pupil'].shape)  # Expected: (batch_size, 3)
    print("Gaze output shape:", outputs['gaze'].shape)      # Expected: (batch_size, 3)