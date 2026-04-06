"""
Swin Transformer + Coordinate Attention 融合模型
作者：轴承故障诊断项目
功能：实现基于Swin Transformer和坐标注意力的轴承故障诊断模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class CoordinateAttention(nn.Module):
    """
    坐标注意力模块
    用于捕获空间位置信息，增强时频特征的空间感知能力
    """
    def __init__(self, in_channels: int, reduction: int = 32):
        super(CoordinateAttention, self).__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        
        x_h = x_h.permute(0, 1, 3, 2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_h * a_w
        
        return out


class WindowAttention(nn.Module):
    """
    窗口多头自注意力机制
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super(WindowAttention, self).__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        
        self.register_buffer("relative_position_index", self._calculate_relative_position_index(window_size))
        
    def _calculate_relative_position_index(self, window_size: int) -> torch.Tensor:
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
            
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer块
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.1):
        super(SwinTransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.window_size = window_size
        self.shift_size = shift_size
        
    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x
        
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # 窗口分割
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # 注意力计算
        attn_windows, _ = self.attn(x_windows, mask_matrix)
        
        # 窗口合并
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # 逆循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H, W, C)
        
        # 残差连接
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x
        
    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows
        
    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x


class PatchMerging(nn.Module):
    """
    补丁合并模块
    """
    def __init__(self, dim: int):
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        # 边界处理
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
            W += 1
            
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class BasicLayer(nn.Module):
    """
    Swin Transformer基础层
    """
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 7):
        super(BasicLayer, self).__init__()
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size, 0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer主干网络
    """
    def __init__(self, 
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 7,
                 in_channels: int = 3):
        super(SwinTransformer, self).__init__()
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = BasicLayer(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size
            )
            self.layers.append(layer)
            
            if i < len(depths) - 1:
                self.layers.append(PatchMerging(embed_dim * (2 ** i)))
        
        self.norm2 = nn.LayerNorm(embed_dim * (2 ** (len(depths) - 1)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 补丁嵌入
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, embed_dim, H/4, W/4)
        x = x.permute(0, 2, 3, 1)  # (B, H/4, W/4, embed_dim)
        x = self.norm1(x)
        
        # Swin Transformer层
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
            else:
                x = layer(x)
                
        x = self.norm2(x)
        
        return x


class SwinCA(nn.Module):
    """
    Swin Transformer + Coordinate Attention 融合模型
    用于轴承故障诊断
    """
    def __init__(self, 
                 num_classes: int = 4,
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 7,
                 in_channels: int = 3,
                 ca_reduction: int = 32):
        super(SwinCA, self).__init__()
        
        # Swin Transformer主干
        self.swin_transformer = SwinTransformer(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            in_channels=in_channels
        )
        
        # Coordinate Attention增强
        self.ca = CoordinateAttention(
            in_channels=embed_dim * (2 ** (len(depths) - 1)),
            reduction=ca_reduction
        )
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * (2 ** (len(depths) - 1)), 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swin Transformer特征提取
        x = self.swin_transformer(x)  # (B, H, W, C)
        
        # 转换为卷积层格式
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Coordinate Attention增强
        x = self.ca(x)
        
        # 全局平均池化
        x = self.avg_pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        
        # 分类
        logits = self.classifier(x)
        
        return logits


def create_swin_ca_model(num_classes: int = 4, 
                      input_channels: int = 3,
                      pretrained: bool = False) -> SwinCA:
    """
    创建Swin-CA模型
    
    Args:
        num_classes (int): 分类类别数
        input_channels (int): 输入通道数
        pretrained (bool): 是否使用预训练权重
        
    Returns:
        SwinCA: Swin-CA模型实例
    """
    model = SwinCA(
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        in_channels=input_channels,
        ca_reduction=32
    )
    
    # 初始化权重
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    model.apply(_init_weights)
    
    return model


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_swin_ca_model(num_classes=4, input_channels=3)
    model = model.to(device)
    
    # 测试输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 64, 2048).to(device)
    
    print(f"输入形状: {input_tensor.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算模型大小
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    model_size = (param_size + buffer_size) / 1024 / 1024
    print(f"模型大小: {model_size:.2f} MB")
    
    print("模型测试完成！")
