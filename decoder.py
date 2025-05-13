import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder as ViTEncoder

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels//32, kernel_size=1)  # Reduced channels
        self.key = nn.Conv2d(channels, channels//32, kernel_size=1)    # Reduced channels
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Always downsample for memory efficiency
        scale = max(1, int((H * W / 1024) ** 0.5))  # More aggressive downsampling
        x_down = F.avg_pool2d(x, kernel_size=scale, stride=scale)
        H_down, W_down = H // scale, W // scale
        
        # Generate query, key, value
        query = self.query(x_down).view(batch_size, -1, H_down*W_down).permute(0, 2, 1)
        key = self.key(x_down).view(batch_size, -1, H_down*W_down)
        value = self.value(x_down).view(batch_size, -1, H_down*W_down)
        
        # Attention with memory-efficient computation
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=2)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H_down, W_down)
        
        # Upsample if needed
        if H_down != H or W_down != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return self.gamma * out + x

class Decoder(nn.Module):
    def __init__(self, image_size=128, patch_size=4, hidden_dim=128, num_heads=4, num_layers=2):  # Reduced dimensions
        super(Decoder, self).__init__()
        
        # Initial processing
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Reduced channels
        
        # Autoencoder part (only 2 downsamples)
        self.autoencoder_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            )
        ])
        
        # ViT part with reduced dimensions
        num_patches = 64  # 8x8
        self.patch_embed = nn.Conv2d(128, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.vit_encoder = ViTEncoder(
            seq_length=num_patches,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=hidden_dim * 2,  # Reduced MLP dimension
            dropout=0.1,
            attention_dropout=0.1
        )
        
        # Decoder part with reduced channels
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, 64, kernel_size=4, stride=2, padding=1),
                ResidualBlock(64)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                ResidualBlock(32)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                ResidualBlock(16)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                ResidualBlock(8)
            )
        ])
        
        # Final layers
        self.final_conv = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Initial processing
        x = self.relu(self.initial_conv(x))
        
        # Autoencoder encoding
        for block in self.autoencoder_blocks:
            x = self.relu(block(x))
        
        # Prepare for ViT
        batch_size = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # ViT processing
        x = self.vit_encoder(x)
        
        # Reshape back to spatial dimensions
        x = x.transpose(1, 2).reshape(batch_size, -1, 8, 8)
        
        # Decoder processing
        for block in self.decoder_blocks:
            x = self.relu(block(x))
        
        # Final processing
        x = self.final_conv(x)
        x = self.tanh(x)
        
        # Guarantee output is 128x128
        if x.shape[-1] != 128 or x.shape[-2] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        
        return x 