import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN encoder using a pretrained ResNet-50 backbone.

    Input:  images of shape (B, 3, H, W), normalized like ImageNet
    Output: feature vectors of shape (B, embed_size)
    """

    def __init__(self, embed_size: int = 256, train_cnn: bool = False):
        super().__init__()

        # 1) Load pretrained ResNet-50 (ImageNet)
        try:
            # Newer torchvision versions
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            # Older versions
            resnet = models.resnet50(pretrained=True)

        # 2) Remove the final classification (fc) layer
        #    Keep everything up to the global average pooling
        modules = list(resnet.children())[:-1]  # all layers except last FC
        self.cnn = nn.Sequential(*modules)

        # 3) Map ResNet feature dimension -> embed_size
        #    ResNet-50 outputs 2048-D features after pooling
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # 4) Optionally freeze CNN weights (pure transfer learning)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W)
        returns: feature vectors of shape (B, embed_size)
        """
        # ResNet body -> (B, 2048, 1, 1)
        features = self.cnn(images)

        # Flatten spatial dimension -> (B, 2048)
        features = features.view(features.size(0), -1)

        # Project to embed_size -> (B, embed_size)
        features = self.fc(features)
        features = self.bn(features)

        return features
