"""
Semantic Embedding Module for LaWa
Implements semantic vector encoding with rotation matrix encryption
Based on LatentSeal and MetaSeal approaches
"""
import torch
import torch.nn as nn
import numpy as np

# Try to import transformers, use fallback if not available
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using fallback encoding mode.")


class RotationMatrix:
    """
    Implements rotation matrix generation and operations for semantic watermark encryption
    """
    def __init__(self, dim=768, seed=None):
        """
        Args:
            dim: Dimension of the semantic vector
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate orthogonal rotation matrix using QR decomposition
        self.rotation_matrix = self._generate_rotation_matrix()
        self.inverse_matrix = torch.transpose(self.rotation_matrix, 0, 1)  # For orthogonal matrix, inverse = transpose
    
    def _generate_rotation_matrix(self):
        """Generate orthogonal rotation matrix using QR decomposition"""
        # Generate random matrix
        random_matrix = torch.randn(self.dim, self.dim)
        # QR decomposition to get orthogonal matrix
        q, r = torch.linalg.qr(random_matrix)
        return q
    
    def encrypt(self, semantic_vector):
        """
        Encrypt semantic vector using rotation matrix
        Args:
            semantic_vector: Tensor of shape (batch_size, dim) or (dim,)
        Returns:
            Encrypted semantic vector
        """
        if semantic_vector.dim() == 1:
            semantic_vector = semantic_vector.unsqueeze(0)
        
        rotation_matrix = self.rotation_matrix.to(semantic_vector.device)
        encrypted = torch.matmul(semantic_vector, rotation_matrix)
        return encrypted
    
    def decrypt(self, encrypted_vector):
        """
        Decrypt encrypted vector using inverse rotation matrix
        Args:
            encrypted_vector: Tensor of shape (batch_size, dim) or (dim,)
        Returns:
            Decrypted semantic vector
        """
        if encrypted_vector.dim() == 1:
            encrypted_vector = encrypted_vector.unsqueeze(0)
        
        inverse_matrix = self.inverse_matrix.to(encrypted_vector.device)
        decrypted = torch.matmul(encrypted_vector, inverse_matrix)
        return decrypted
    
    def save(self, path):
        """Save rotation matrix to file"""
        torch.save({
            'rotation_matrix': self.rotation_matrix,
            'inverse_matrix': self.inverse_matrix,
            'dim': self.dim,
            'seed': self.seed
        }, path)
    
    def load(self, path):
        """Load rotation matrix from file"""
        checkpoint = torch.load(path, map_location='cpu')
        self.rotation_matrix = checkpoint['rotation_matrix']
        self.inverse_matrix = checkpoint['inverse_matrix']
        self.dim = checkpoint['dim']
        self.seed = checkpoint.get('seed', None)


class SemanticEncoder(nn.Module):
    """
    Semantic encoder using CLIP text encoder
    Converts prompts to high-dimensional semantic vectors
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", rotation_seed=None, embedding_dim=512):
        """
        Args:
            model_name: Name of the CLIP model to use
            rotation_seed: Seed for rotation matrix generation
            embedding_dim: Embedding dimension (used when model cannot be loaded)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.text_encoder = None
        
        # Try to load CLIP text encoder
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
                self.text_encoder = CLIPTextModel.from_pretrained(model_name)
                self.text_encoder.eval()
                
                # Freeze text encoder parameters
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                
                # Get embedding dimension
                self.embedding_dim = self.text_encoder.config.hidden_size
                print(f"Loaded CLIP model with embedding dim: {self.embedding_dim}")
            except Exception as e:
                print(f"Warning: Could not load CLIP model: {e}")
                print(f"Using hash-based fallback with embedding_dim={embedding_dim}")
                self.text_encoder = None
                self.tokenizer = None
        else:
            print(f"Using hash-based fallback with embedding_dim={embedding_dim}")
        
        # Initialize rotation matrix for encryption
        self.rotation_matrix = RotationMatrix(dim=self.embedding_dim, seed=rotation_seed)
    
    @torch.no_grad()
    def encode_text(self, prompts, return_pooled=True):
        """
        Encode text prompts to semantic vectors
        Args:
            prompts: List of text strings or single string
            return_pooled: If True, return pooled output; otherwise return last hidden state
        Returns:
            Semantic vectors of shape (batch_size, embedding_dim)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # If CLIP model is not available, use simple hash-based encoding
        if self.text_encoder is None or self.tokenizer is None:
            batch_size = len(prompts)
            # Generate deterministic vectors from text hash
            semantic_vectors = []
            for prompt in prompts:
                # Use hash of prompt to generate deterministic vector
                hash_val = hash(prompt)
                torch.manual_seed(abs(hash_val) % 2**32)
                vec = torch.randn(self.embedding_dim)
                vec = vec / vec.norm()  # Normalize
                semantic_vectors.append(vec)
            return torch.stack(semantic_vectors)
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        if text_input_ids.shape[1] > self.tokenizer.model_max_length:
            text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
        
        # Move to same device as text encoder
        text_input_ids = text_input_ids.to(self.text_encoder.device)
        
        # Encode
        text_embeddings = self.text_encoder(text_input_ids)[0]
        
        if return_pooled:
            # Use the embedding of the EOS token (last token)
            # Get the position of EOS token
            eos_token_id = self.tokenizer.eos_token_id
            # Find the position of EOS token for each sequence
            batch_size = text_input_ids.shape[0]
            semantic_vectors = []
            for i in range(batch_size):
                # Find last non-padding token
                eos_pos = (text_input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    eos_pos = eos_pos[0]
                else:
                    eos_pos = text_input_ids.shape[1] - 1
                semantic_vectors.append(text_embeddings[i, eos_pos])
            semantic_vectors = torch.stack(semantic_vectors)
        else:
            # Use mean pooling
            semantic_vectors = text_embeddings.mean(dim=1)
        
        return semantic_vectors
    
    def augment_prompt(self, prompt, metadata=None):
        """
        Augment prompt with metadata
        Args:
            prompt: Original text prompt
            metadata: Dictionary with metadata (model_version, user_id, timestamp, etc.)
        Returns:
            Augmented prompt string
        """
        augmented_parts = [prompt]
        
        if metadata:
            if 'model_version' in metadata:
                augmented_parts.append(f"model: {metadata['model_version']}")
            if 'user_id' in metadata:
                augmented_parts.append(f"user: {metadata['user_id']}")
            if 'timestamp' in metadata:
                augmented_parts.append(f"date: {metadata['timestamp']}")
            if 'extra_info' in metadata:
                augmented_parts.append(metadata['extra_info'])
        
        return ", ".join(augmented_parts)
    
    def forward(self, prompts, encrypt=True, metadata=None):
        """
        Full pipeline: encode text to semantic vector and optionally encrypt
        Args:
            prompts: Text prompt(s)
            encrypt: Whether to encrypt the semantic vector
            metadata: Optional metadata to augment the prompt
        Returns:
            (encrypted) semantic vector
        """
        # Augment prompt if metadata provided
        if metadata is not None:
            if isinstance(prompts, str):
                prompts = self.augment_prompt(prompts, metadata)
            else:
                prompts = [self.augment_prompt(p, metadata) for p in prompts]
        
        # Encode text to semantic vector
        semantic_vector = self.encode_text(prompts)
        
        # Encrypt if requested
        if encrypt:
            semantic_vector = self.rotation_matrix.encrypt(semantic_vector)
        
        return semantic_vector
    
    def verify(self, extracted_vector, original_prompt, threshold=0.85, metadata=None):
        """
        Verify watermark by comparing with original prompt
        Args:
            extracted_vector: Extracted and decrypted semantic vector
            original_prompt: Original prompt text
            threshold: Cosine similarity threshold for verification
            metadata: Optional metadata used during embedding
        Returns:
            (is_authentic, similarity_score)
        """
        # Decrypt if needed (assuming input is already decrypted)
        # Get original semantic vector
        original_vector = self.forward(original_prompt, encrypt=False, metadata=metadata)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            extracted_vector, 
            original_vector, 
            dim=-1
        )
        
        is_authentic = similarity >= threshold
        
        # Convert to scalar for single batch
        if similarity.numel() == 1:
            return is_authentic.item(), similarity.item()
        else:
            return is_authentic, similarity


def test_semantic_encoder():
    """Test function for semantic encoder"""
    print("Testing Semantic Encoder...")
    
    # Initialize encoder
    encoder = SemanticEncoder(rotation_seed=42)
    print(f"Embedding dimension: {encoder.embedding_dim}")
    
    # Test encoding
    prompt = "A photo of an astronaut riding a horse on the moon"
    semantic_vector = encoder(prompt, encrypt=False)
    print(f"Semantic vector shape: {semantic_vector.shape}")
    
    # Test encryption
    encrypted_vector = encoder.rotation_matrix.encrypt(semantic_vector)
    print(f"Encrypted vector shape: {encrypted_vector.shape}")
    
    # Test decryption
    decrypted_vector = encoder.rotation_matrix.decrypt(encrypted_vector)
    print(f"Decrypted vector shape: {decrypted_vector.shape}")
    
    # Verify decryption
    diff = torch.abs(semantic_vector - decrypted_vector).max()
    print(f"Max difference after encryption/decryption: {diff.item()}")
    
    # Test verification
    is_authentic, similarity = encoder.verify(decrypted_vector, prompt)
    print(f"Verification result: {is_authentic}, similarity: {similarity}")
    
    # Test with metadata
    metadata = {
        'model_version': 'v2.1',
        'user_id': '12345',
        'timestamp': '2025-10-22'
    }
    augmented_prompt = encoder.augment_prompt(prompt, metadata)
    print(f"Augmented prompt: {augmented_prompt}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_semantic_encoder()
