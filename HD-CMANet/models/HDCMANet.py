import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.PSConv import PSConv
from modules.CMCA import CMCAttn
from modules.AHFF import AHFF


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False):
        super().__init__()
        assert isinstance(dim, int), f"dim must be an integer, got {type(dim)}"
        assert isinstance(mlp_ratio, (int, float)), f"mlp_ratio must be an int or float, got {type(mlp_ratio)}"
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x, x, x)[0]
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class ViTBranch(nn.Module):

    def __init__(self, in_channels, dim=64, depth=3, num_heads=8, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (256 // patch_size) ** 2, dim))  # 假设输入为256x256
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads) for _ in range(depth)
        ])
        self.up = nn.Upsample(scale_factor=patch_size, mode='bilinear')

    def forward(self, x):
        x = self.patch_embed(x)
        B, D, Hp, Wp = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed[:, :Hp * Wp]

        for blk in self.blocks:
            x = blk(x)

        x = rearrange(x, 'b (h w) d -> b d h w', h=Hp, w=Wp)
        x = self.up(x)
        return x


class HDCMANet(nn.Module):
    def __init__(self, arch, scale_ratio, n_select_bands, n_bands, dim=64, dataset=None):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        # 第一阶段CNN
        self.conv_fus_cnn = nn.Sequential(
            nn.Conv2d(n_bands, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        # 第一阶段ViT
        self.conv_fus_vit = ViTBranch(n_bands, dim)
        self.cross_attn1 = CMCAttn(dim)  # 第一次跨模态融合

        # 第二阶段CNN
        self.cnn_branch = nn.Sequential(
            PSConv(dim, dim, k=3),
            nn.ReLU(),
            PSConv(dim, dim, k=3)
        )
        # 第二阶段ViT
        self.vit_branch = ViTBranch(dim, dim)
        self.cross_attn2 = CMCAttn(dim)  # 第二次跨模态融合

        # 自适应层次化特征融合策略
        self.triple_fusion = AHFF(dim=64, dataset=dataset)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, n_bands, 1)
        )
        self.conv_spat = nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1)
        self.conv_spec = nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1)

    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands-1.0)
        for i in range(0, self.n_select_bands-1):
            x_lr[:, int(gap_bands*i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands-1), ::] = x_hr[:, self.n_select_bands-1, ::]

        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

        return edge

    def forward(self, x_lr, x_hr):
        x = self.lrhr_interpolate(x_lr, x_hr)

        # 第一阶段处理
        x_cnn = self.conv_fus_cnn(x)  # CNN路径
        x_vit = self.conv_fus_vit(x)  # ViT路径
        stage1 = self.cross_attn1(x_cnn, x_vit)  # 第一阶段交互结果

        # 第二阶段处理
        cnn_feat = self.cnn_branch(x_cnn)  # 处理原始CNN特征
        vit_feat = self.vit_branch(x_vit)  # 处理原始ViT特征
        stage2 = self.cross_attn2(cnn_feat, vit_feat)  # 第二阶段交互结果

        # 双分支拼接（使用未交互特征）
        concat_feat = torch.cat([cnn_feat, vit_feat], dim=1)
        stage3 = self.fusion(concat_feat)

        # 自适应层次化特征融合策略
        fused = self.triple_fusion(stage1, stage2, stage3)

        x_spat = fused + self.conv_spat(fused)
        spat_edge1, spat_edge2 = self.spatial_edge(x_spat)

        x_spec = x_spat + self.conv_spec(x_spat)
        spec_edge = self.spectral_edge(x_spec)
        x = x_spec

        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge


