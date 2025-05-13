import torch
import torch.nn as nn
import torch.nn.functional as F

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
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels//16, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//16, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Downsample for memory efficiency
        if H * W > 4096:  # Limit attention computation for large feature maps
            scale = int((H * W / 4096) ** 0.5)
            x_down = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            H_down, W_down = H // scale, W // scale
        else:
            x_down = x
            H_down, W_down = H, W
        
        # Generate query, key, value
        query = self.query(x_down).view(batch_size, -1, H_down*W_down).permute(0, 2, 1)
        key = self.key(x_down).view(batch_size, -1, H_down*W_down)
        value = self.value(x_down).view(batch_size, -1, H_down*W_down)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=2)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H_down, W_down)
        
        # Upsample if needed
        if H_down != H or W_down != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return self.gamma * out + x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Input: 3 channels (for encrypted cover image)
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # Encoder blocks with residual connections
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(64),
                AttentionBlock(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(128),
                AttentionBlock(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(256),
                AttentionBlock(256),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
            )
        ])
        
        # Final layers
        self.final_blocks = nn.Sequential(
            ResidualBlock(512),
            AttentionBlock(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1)
        )
        
        # Upsampling path: 16x16 -> 32x32 -> 64x64 -> 128x128
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                ResidualBlock(128),
                AttentionBlock(128)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                ResidualBlock(64),
                AttentionBlock(64)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                ResidualBlock(32),
                AttentionBlock(32)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                ResidualBlock(16),
                AttentionBlock(16)
            )
        ])
        
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.initial_conv(x))
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = self.relu(block(x))
        
        # Final processing
        x = self.final_blocks(x)
        
        # Upsampling path
        for block in self.upsample_blocks:
            x = self.relu(block(x))
        
        # Final processing
        x = self.final_conv(x)
        x = self.tanh(x)
        
        if x.shape[-1] != 128 or x.shape[-2] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        
        return x