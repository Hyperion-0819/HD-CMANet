import torch
import torch.nn as nn


class AHFF(nn.Module):

    def __init__(self, dim, dataset):
        global n_bands
        super().__init__()
        self.dataset_bands = {
            'Pavia': 102,
            'PaviaU': 103,
            'Botswana': 145,
            'KSC': 176,
            'Urban': 162,
            'IndianP': 200,
            'Washington': 191,
            'MUUFL_HSI': 64,
            'Houston_HSI': 144,
            'salinas_corrected': 204,
            'Chikusei': 128
        }
        self.n_bands = self.dataset_bands.get(dataset, 103)
        self.channel_adjust = nn.Conv2d(64, self.n_bands, 1)
        self.fusion_net = nn.Sequential(
            nn.Conv2d(self.n_bands * 3, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, self.n_bands, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, stage1, stage2, stage3):
        stage1 = self.channel_adjust(stage1)
        stage2 = self.channel_adjust(stage2)
        fused = torch.cat([stage1, stage2, stage3], dim=1)
        weight_map = self.fusion_net(fused)
        return weight_map * stage1 + (1 - weight_map) * stage2 + stage3