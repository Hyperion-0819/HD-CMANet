import torch
import torch.nn as nn
from einops import rearrange
from modules.CLA import CLAttention

class CMCAttn(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.cla_cnn = CLAttention(channels)  # CNN分支的通道局部注意力
        self.cla_vit = CLAttention(channels)  # ViT分支的通道局部注意力

        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Linear(channels, channels // 8)
        self.value_conv = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, cnn_feat, vit_feat):
        # 输入特征预处理
        cnn_att = self.cla_cnn(cnn_feat)
        vit_att = self.cla_vit(vit_feat)
        batch, C, H, W = cnn_feat.size()

        # 处理CNN特征作为query
        proj_query = self.query_conv(cnn_att).view(batch, -1, H * W).permute(0, 2, 1)

        # 处理ViT特征作为key/value
        vit_att = rearrange(vit_att, 'b c h w -> b (h w) c')
        proj_key = self.key_conv(vit_att)
        proj_value = self.value_conv(vit_att)

        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        attention = self.softmax(energy)
        out = torch.bmm(attention, proj_value)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

        return self.gamma * out + cnn_feat