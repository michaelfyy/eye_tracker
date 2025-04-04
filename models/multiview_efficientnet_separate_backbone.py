import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiviewEfficientNetSB(nn.Module):
    def __init__(self, 
                 num_views: int = 4, 
                 pretrained: bool = True):
        super(MultiviewEfficientNetSB, self).__init__()
        # Load a pretrained EfficientNetV2-M model from torchvision
        weights = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        self.base1 = models.efficientnet_v2_m(weights=weights)
        self.base2 = models.efficientnet_v2_m(weights=weights)
        self.base3 = models.efficientnet_v2_m(weights=weights)
        self.base4 = models.efficientnet_v2_m(weights=weights)
        
        # Obtain the number of features from the current classifier layer.
        # EfficientNetV2-M classifier is typically a Sequential with a Dropout and a Linear layer.
        in_features = self.base1.classifier[1].in_features
        concat_features = in_features * num_views
        
        # Remove the original classifier by replacing it with an identity module.
        self.base1.classifier = nn.Identity()
        self.base2.classifier = nn.Identity()
        self.base3.classifier = nn.Identity()
        self.base4.classifier = nn.Identity()
        
        # Define a regression head for 2D localization. Output 2D vector: [pupil_x, pupil_y] in image space
        self.regressor_2d_cam_1 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.regressor_2d_cam_2 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.regressor_2d_cam_3 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.regressor_2d_cam_4 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        # Define a regression head for 3D localization. Output 6D vector: [pupil_x, y, z, gaze_endpoint_x, y, z]
        self.regressor_3d_cam_1 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )

        self.regressor_3d_cam_2 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )

        self.regressor_3d_cam_3 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )

        self.regressor_3d_cam_4 = nn.Sequential(
            nn.Linear(concat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )
        
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch, views, channels, height, width)
            
        Returns:
            dict: A dictionary with keys containing the respective regression outputs.
        """
        batch, views, channels, height, width = x.shape

        # Reshape to combine batch and view dimensions.
        x1 = x[:,0,...]
        x2 = x[:,1,...]
        x3 = x[:,2,...]
        x4 = x[:,3,...]
        
        # Pass all views through the feature extractor.
        features1 = self.base1(x1)
        features2 = self.base2(x2)
        features3 = self.base3(x3)
        features4 = self.base4(x4)
        
        # Concatenate features from all views along the feature dimension.
        features_cat = torch.cat((features1, features2, features3, features4), dim=1)
        
        # Pass concatenated features through both regression heads.
        cam_1_2d = self.regressor_2d_cam_1(features_cat)
        cam_2_2d = self.regressor_2d_cam_2(features_cat)
        cam_3_2d = self.regressor_2d_cam_3(features_cat)
        cam_4_2d = self.regressor_2d_cam_4(features_cat)
        cam_1_3d = self.regressor_3d_cam_1(features_cat)
        cam_2_3d = self.regressor_3d_cam_2(features_cat)
        cam_3_3d = self.regressor_3d_cam_3(features_cat)
        cam_4_3d = self.regressor_3d_cam_4(features_cat)
        
        return {'cam_1_2d': cam_1_2d, 'cam_2_2d': cam_2_2d, 'cam_3_2d': cam_3_2d, 'cam_4_2d': cam_4_2d,
                'cam_1_3d': cam_1_3d, 'cam_2_3d': cam_2_3d, 'cam_3_3d': cam_3_3d, 'cam_4_3d': cam_4_3d}

if __name__ == "__main__":
    # Quick test to ensure the model runs correctly.
    # Let's assume we have 4 views per sample and each image is 3x224x224.
    num_views = 4
    batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiviewEfficientNetSB(num_views=num_views, pretrained=False).to(device)
    # Create a dummy input of shape (batch, views, channels, height, width).
    dummy_input = torch.randn(batch_size, num_views, 3, 224, 224).to(device)  # Batch of samples
    outputs = model(dummy_input)
    print("2D regressor output shape:", outputs['cam_1_2d'].shape)  # Expected: (batch_size, 4)
    print("3D regressor output shape:", outputs['cam_1_3d'].shape)  # Expected: (batch_size, 6)