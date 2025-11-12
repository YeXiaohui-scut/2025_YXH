"""
Utility functions for Hybrid Watermarking Framework
Combines MetaSeal, SWIFT, and GenPTW approaches
"""

import os
import io
import json
import hashlib
from datetime import datetime
from typing import Tuple, Dict, Optional, Any
import warnings

import torch
import numpy as np
from PIL import Image
import qrcode
from pyzbar.pyzbar import decode as decode_qr

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature


def load_semantic_extractor(device: str = 'cuda'):
    """
    Load BLIP model for image captioning (SWIFT approach)
    
    Args:
        device: 'cuda' or 'cpu'
        
    Returns:
        tuple: (model, processor) for BLIP
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print("Loading BLIP model for semantic extraction...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        model.eval()
        
        print("✓ BLIP model loaded successfully")
        return model, processor
        
    except Exception as e:
        print(f"Error loading BLIP: {e}")
        print("Please install transformers: pip install transformers")
        raise


def extract_semantic_description(image: Image.Image, model, processor, device: str = 'cuda') -> str:
    """
    Extract text description from image using BLIP
    
    Args:
        image: PIL Image
        model: BLIP model
        processor: BLIP processor
        device: device to run on
        
    Returns:
        str: Text description
    """
    with torch.no_grad():
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


def generate_watermark_qr(
    text_data: str,
    user_id: str,
    timestamp: Optional[str] = None,
    private_key_path: str = "keys/private_key.pem"
) -> torch.Tensor:
    """
    Generate QR code watermark with cryptographic signature (MetaSeal approach)
    
    Args:
        text_data: Semantic description or message
        user_id: User identifier
        timestamp: ISO timestamp (if None, uses current time)
        private_key_path: Path to ECDSA private key
        
    Returns:
        torch.Tensor: Binary QR code image [1, 256, 256]
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Create message dictionary
    message_dict = {
        'text_data': text_data,
        'user_id': user_id,
        'timestamp': timestamp
    }
    message_str = json.dumps(message_dict, sort_keys=True)
    message_bytes = message_str.encode('utf-8')
    
    # Load private key
    if not os.path.exists(private_key_path):
        print(f"Warning: Private key not found at {private_key_path}")
        print("Generating new key pair...")
        private_key, public_key = generate_key_pair()
        
        # Save keys
        os.makedirs(os.path.dirname(private_key_path) or '.', exist_ok=True)
        save_private_key(private_key, private_key_path)
        
        public_key_path = private_key_path.replace('private', 'public')
        save_public_key(public_key, public_key_path)
    else:
        private_key = load_private_key(private_key_path)
    
    # Sign message with ECDSA (P-256)
    signature = private_key.sign(
        message_bytes,
        ec.ECDSA(hashes.SHA256())
    )
    
    # Combine message and signature
    combined_data = {
        'message': message_str,
        'signature': signature.hex()
    }
    combined_str = json.dumps(combined_data)
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=None,  # Auto-determine version
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,
        border=4,
    )
    qr.add_data(combined_str)
    qr.make(fit=True)
    
    # Convert to PIL Image
    qr_image = qr.make_image(fill_color="black", back_color="white")
    
    # Resize to 256x256 and convert to tensor
    qr_image = qr_image.resize((256, 256), Image.Resampling.LANCZOS)
    qr_array = np.array(qr_image)
    
    # Convert to binary (0 or 1)
    qr_binary = (qr_array < 128).astype(np.float32)
    
    # Convert to torch tensor [1, 256, 256]
    qr_tensor = torch.from_numpy(qr_binary).unsqueeze(0)
    
    return qr_tensor


def verify_watermark_qr(
    qr_image_tensor: torch.Tensor,
    public_key_path: str = "keys/public_key.pem"
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify QR code watermark signature (MetaSeal verification)
    
    Args:
        qr_image_tensor: Extracted QR code [1, 256, 256] or [256, 256]
        public_key_path: Path to ECDSA public key
        
    Returns:
        tuple: (is_valid, message_dict) where message_dict contains decoded data
    """
    try:
        # Convert tensor to PIL Image
        if qr_image_tensor.dim() == 3:
            qr_image_tensor = qr_image_tensor[0]
        
        qr_array = (qr_image_tensor.cpu().numpy() * 255).astype(np.uint8)
        qr_image = Image.fromarray(qr_array, mode='L')
        
        # Decode QR code
        decoded_objects = decode_qr(qr_image)
        
        if not decoded_objects:
            print("Failed to decode QR code")
            return False, None
        
        # Get data from QR code
        qr_data = decoded_objects[0].data.decode('utf-8')
        combined_data = json.loads(qr_data)
        
        message_str = combined_data['message']
        signature_hex = combined_data['signature']
        signature = bytes.fromhex(signature_hex)
        
        # Load public key
        if not os.path.exists(public_key_path):
            print(f"Warning: Public key not found at {public_key_path}")
            return False, None
        
        public_key = load_public_key(public_key_path)
        
        # Verify signature
        try:
            public_key.verify(
                signature,
                message_str.encode('utf-8'),
                ec.ECDSA(hashes.SHA256())
            )
            
            # Parse message
            message_dict = json.loads(message_str)
            
            return True, message_dict
            
        except InvalidSignature:
            print("Invalid signature - watermark has been tampered!")
            return False, None
            
    except Exception as e:
        print(f"Error verifying watermark: {e}")
        return False, None


def load_t2i_model(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = 'cuda'):
    """
    Load Stable Diffusion T2I model
    
    Args:
        model_id: HuggingFace model ID
        device: 'cuda' or 'cpu'
        
    Returns:
        Stable Diffusion pipeline
    """
    try:
        from diffusers import StableDiffusionPipeline
        
        print(f"Loading T2I model: {model_id}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        pipe = pipe.to(device)
        pipe.safety_checker = None  # Disable for research
        
        print("✓ T2I model loaded successfully")
        return pipe
        
    except Exception as e:
        print(f"Error loading T2I model: {e}")
        print("Please install diffusers: pip install diffusers")
        raise


# ============================================================================
# Cryptographic Key Management
# ============================================================================

def generate_key_pair() -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    """Generate ECDSA key pair (P-256 curve)"""
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key


def save_private_key(private_key: ec.EllipticCurvePrivateKey, path: str):
    """Save private key to PEM file"""
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open(path, 'wb') as f:
        f.write(pem)
    print(f"Private key saved to {path}")


def save_public_key(public_key: ec.EllipticCurvePublicKey, path: str):
    """Save public key to PEM file"""
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open(path, 'wb') as f:
        f.write(pem)
    print(f"Public key saved to {path}")


def load_private_key(path: str) -> ec.EllipticCurvePrivateKey:
    """Load private key from PEM file"""
    with open(path, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )
    return private_key


def load_public_key(path: str) -> ec.EllipticCurvePublicKey:
    """Load public key from PEM file"""
    with open(path, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )
    return public_key


# ============================================================================
# Image Processing Utilities
# ============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    tensor = tensor.clamp(0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, device: str = 'cuda') -> torch.Tensor:
    """Convert PIL Image to torch tensor"""
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image file"""
    image = tensor_to_pil(tensor)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    image.save(path)
    print(f"Image saved to {path}")


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts
    Simple implementation using word overlap; can be enhanced with BERT
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score [0, 1]
    """
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def compute_bert_similarity(text1: str, text2: str, device: str = 'cuda') -> float:
    """
    Compute BERT-based semantic similarity (enhanced version)
    
    Args:
        text1: First text
        text2: Second text
        device: Computing device
        
    Returns:
        float: Cosine similarity score [-1, 1]
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        
        # Load BERT model (can be cached)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        model.eval()
        
        # Encode texts
        with torch.no_grad():
            inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True).to(device)
            inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True).to(device)
            
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
            
            # Use CLS token embedding
            emb1 = outputs1.last_hidden_state[:, 0, :]
            emb2 = outputs2.last_hidden_state[:, 0, :]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
        
        return similarity
        
    except Exception as e:
        print(f"BERT similarity failed, falling back to word overlap: {e}")
        return compute_similarity(text1, text2)


# ============================================================================
# Testing and Demo Functions
# ============================================================================

def test_qr_generation_and_verification():
    """Test QR code generation and verification pipeline"""
    print("\n" + "="*60)
    print("Testing QR Code Generation and Verification")
    print("="*60)
    
    # Generate keys
    os.makedirs('keys', exist_ok=True)
    private_key, public_key = generate_key_pair()
    save_private_key(private_key, 'keys/private_key.pem')
    save_public_key(public_key, 'keys/public_key.pem')
    
    # Generate watermark
    text_data = "A beautiful sunset over the ocean"
    user_id = "user_12345"
    timestamp = "2025-11-12T12:00:00"
    
    print(f"\nGenerating watermark for:")
    print(f"  Text: {text_data}")
    print(f"  User: {user_id}")
    print(f"  Time: {timestamp}")
    
    qr_tensor = generate_watermark_qr(text_data, user_id, timestamp, 'keys/private_key.pem')
    print(f"✓ Generated QR tensor: {qr_tensor.shape}")
    
    # Verify watermark
    print("\nVerifying watermark...")
    is_valid, message = verify_watermark_qr(qr_tensor, 'keys/public_key.pem')
    
    if is_valid:
        print("✓ Signature is VALID!")
        print(f"  Decoded message: {message}")
    else:
        print("✗ Signature is INVALID!")
    
    # Test tampering detection
    print("\nTesting tampering detection...")
    tampered_tensor = qr_tensor.clone()
    tampered_tensor[0, 100:110, 100:110] = 1 - tampered_tensor[0, 100:110, 100:110]
    
    is_valid_tampered, _ = verify_watermark_qr(tampered_tensor, 'keys/public_key.pem')
    
    if not is_valid_tampered:
        print("✓ Tampering detected successfully!")
    else:
        print("✗ Failed to detect tampering")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    print("Hybrid Watermarking Framework - Utils Module")
    print("Testing core functionality...\n")
    
    test_qr_generation_and_verification()
    
    print("\n✓ All utility functions are ready!")
    print("\nNext steps:")
    print("1. Install required packages: pip install -r requirements_watermark.txt")
    print("2. Implement models.py")
    print("3. Implement train.py")
    print("4. Implement main.py")
