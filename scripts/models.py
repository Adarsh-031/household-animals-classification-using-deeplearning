import torch
from torch import nn


class VGG6(nn.Module):
    """
    Architecture following Figure 3 of Lei Lin's paper.

    Unspecified settings chosen for this implementation:
    - Input size: 224 x 224
    - Max pooling: 2 x 2, stride 2
    - Hidden dense layer: 512 units
    - Binary output: one logit
    """

    def __init__(self, image_size=224, hidden_units=512, dropout=0.0):
        super().__init__()

        if image_size < 16 or image_size % 16 != 0:
            raise ValueError("image_size must be a positive multiple of 16.")

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        feature_size = image_size // 16

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(128 * feature_size * feature_size, hidden_units),
            nn.ReLU(),
        ]

        if dropout > 0:
            classifier_layers.append(nn.Dropout(p=dropout))

        classifier_layers.append(nn.Linear(hidden_units, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features).squeeze(1)

class VGG16Binary(nn.Module):
    """
    Pretrained VGG-16 convolutional features plus a binary classifier.

    Classifier settings are implementation choices because the
    reference paper does not fully specify its replacement head.
    """

    def __init__(self, hidden_units=512, dropout=0.5, pretrained=True):
        super().__init__()

        from torchvision.models import vgg16, VGG16_Weights

        weights = (
            VGG16_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = vgg16(weights=weights)

        # Retain the pretrained convolutional feature extractor.
        # Discard VGG-16's original ImageNet classifier.
        self.features = backbone.features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, 1),
        )

        # Stage 1: train only the replacement classifier.
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features).squeeze(1)