import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from utils.encryption import encrypt_image, decrypt_image

class SteganographyNet(nn.Module):
    def __init__(self, encryption_key=42):
        super(SteganographyNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encryption_key = encryption_key
        
    def forward(self, cover_image, secret_image):
        # Encrypt the cover image
        encrypted_cover = encrypt_image(cover_image, self.encryption_key)
        
        # Generate stego image
        stego_image = self.encoder(encrypted_cover)
        
        # Decrypt the stego image
        decrypted_stego = decrypt_image(stego_image, self.encryption_key)
        
        # Recover secret image
        recovered_secret = self.decoder(decrypted_stego)
        
        return stego_image, recovered_secret 