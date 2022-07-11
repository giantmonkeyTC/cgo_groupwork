from typing import Tuple
import torch
from torch import nn

class SegNetBasicEncoder(nn.Module):
    def __init__(self, n_channels=3, bn_momentum=0.1):
        """
        n_channels: number of channels (default=3)
        bn_momentum: momentum of batch normalization (default=0.1)
        """
        super().__init__()

        self.down_pooling = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80, momentum=bn_momentum),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(80, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96, momentum=bn_momentum),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, input: torch.Tensor):
        x = self.conv1(input)
        x, idx1 = self.down_pooling(x)
        size1 = x.size()

        x = self.conv2(x)
        x, idx2 = self.down_pooling(x)
        size2 = x.size()

        x = self.conv3(x)
        x, idx3 = self.down_pooling(x)
        size3 = x.size()

        x = self.conv4(x)

        return x, (idx1, idx2, idx3), (size1, size2, size3)

class SegNetBasicDecoder(nn.Module):
    def __init__(self, n_out=3, bn_momentum=0.1):
        """
        n_out: number of out channels (default=3)
        bn_momentum: momentum of batch normalization (default=0.1)
        """
        super().__init__()

        self.up_pooling = nn.MaxUnpool2d(2, stride=2)

        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96, momentum=bn_momentum),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(96, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80, momentum=bn_momentum),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(80, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.Conv2d(64, n_out, kernel_size=3, padding=1)
        )

    def forward(
        self, 
        input: torch.Tensor, 
        pooling_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pooling_sizes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x = self.deconv1(input)

        x = self.up_pooling(x, pooling_indices[2], output_size=pooling_sizes[1])
        x = self.deconv2(x)

        x = self.up_pooling(x, pooling_indices[1], output_size=pooling_sizes[0])
        x = self.deconv3(x)

        x = self.up_pooling(x, pooling_indices[0])
        x = self.deconv4(x)

        return x

class SegNetBasic(nn.Module):
    def __init__(self, n_channels=3, n_out=3, bn_momentum=0.1):
        """
        n_channels: number of channels (default=3)
        bn_momentum: momentum of batch normalization (default=0.1)
        """
        super().__init__()

        self.encoder = SegNetBasicEncoder(n_channels, bn_momentum)
        self.decoder = SegNetBasicDecoder(n_out, bn_momentum)

    def forward(self, input: torch.Tensor):
        z, pooling_indices, pooling_sizes = self.encoder(input)
        y = self.decoder(z, pooling_indices, pooling_sizes)

        return y


if __name__ == "__main__":
    image = torch.randn((1, 3, 352, 1216)).float()

    segnet = SegNetBasic()

    print(segnet(image).shape)