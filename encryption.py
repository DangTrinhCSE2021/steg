import torch
import numpy as np

def generate_key_tensor(shape, key=42):
    """Generate a random key tensor with the same shape as the input image."""
    np.random.seed(key)
    key_tensor = torch.from_numpy(np.random.randint(0, 256, shape, dtype=np.uint8))
    return key_tensor

def encrypt_image(image_tensor, key=42):
    """
    Encrypt the image using XOR operation with a random key.
    Args:
        image_tensor: Input tensor in range [0, 1]
        key: Random seed for key generation
    Returns:
        Encrypted tensor in range [0, 1]
    """
    # Convert to integer values (0-255)
    image_int = (image_tensor * 255).to(torch.uint8)
    
    # Generate key tensor
    key_tensor = generate_key_tensor(image_int.shape, key)
    
    # Apply XOR operation
    encrypted_int = torch.bitwise_xor(image_int, key_tensor)
    
    # Convert back to float values [0, 1]
    encrypted_float = encrypted_int.float() / 255.0
    
    return encrypted_float

def decrypt_image(encrypted_tensor, key=42):
    """
    Decrypt the image using XOR operation with the same key.
    Args:
        encrypted_tensor: Encrypted tensor in range [0, 1]
        key: Random seed for key generation (must be same as encryption)
    Returns:
        Decrypted tensor in range [0, 1]
    """
    # Convert to integer values (0-255)
    encrypted_int = (encrypted_tensor * 255).to(torch.uint8)
    
    # Generate key tensor (same as encryption)
    key_tensor = generate_key_tensor(encrypted_int.shape, key)
    
    # Apply XOR operation
    decrypted_int = torch.bitwise_xor(encrypted_int, key_tensor)
    
    # Convert back to float values [0, 1]
    decrypted_float = decrypted_int.float() / 255.0
    
    return decrypted_float 