# === COMPREHENSIVE ALGORITHMS PART 2 - ADVANCED NEURAL NETWORKS ===
# Focus: Advanced pattern recognition for maximum diversity before edge of chaos training

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import defaultdict

# === VISION TRANSFORMER VARIANTS ===

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block for hierarchical pattern recognition"""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)
        
    def forward(self, x, mask=None):
        H, W = x.shape[1], x.shape[2]
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x

class WindowAttention(nn.Module):
    """Window-based multi-head attention"""
    
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            attn = attn + mask.unsqueeze(1)
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class EfficientNet(nn.Module):
    """EfficientNet for efficient pattern recognition"""
    
    def __init__(self, width_coefficient: float = 1.0, depth_coefficient: float = 1.0, 
                 resolution: int = 224, num_classes: int = 1000):
        super().__init__()
        
        # Calculate scaled dimensions
        def round_filters(filters, width_coefficient):
            return int(filters * width_coefficient)
        
        def round_repeats(repeats, depth_coefficient):
            return int(math.ceil(repeats * depth_coefficient))
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, round_filters(32, width_coefficient), 3, 2, 1, bias=False),
            nn.BatchNorm2d(round_filters(32, width_coefficient)),
            nn.SiLU(inplace=True)
        )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        
        # Block configurations: (kernel_size, stride, expand_ratio, input_filters, output_filters, num_repeats)
        block_configs = [
            (3, 1, 1, 32, 16, 1),
            (3, 2, 6, 16, 24, 2),
            (5, 2, 6, 24, 40, 2),
            (3, 2, 6, 40, 80, 3),
            (5, 1, 6, 80, 112, 3),
            (5, 2, 6, 112, 192, 4),
            (3, 1, 6, 192, 320, 1),
        ]
        
        for kernel_size, stride, expand_ratio, input_filters, output_filters, num_repeats in block_configs:
            input_filters = round_filters(input_filters, width_coefficient)
            output_filters = round_filters(output_filters, width_coefficient)
            num_repeats = round_repeats(num_repeats, depth_coefficient)
            
            # First block
            self.blocks.append(MBConvBlock(input_filters, output_filters, kernel_size, stride, expand_ratio))
            
            # Remaining blocks
            for _ in range(num_repeats - 1):
                self.blocks.append(MBConvBlock(output_filters, output_filters, kernel_size, 1, expand_ratio))
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(round_filters(320, width_coefficient), round_filters(1280, width_coefficient), 1, bias=False),
            nn.BatchNorm2d(round_filters(1280, width_coefficient)),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(round_filters(1280, width_coefficient), num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv Block"""
    
    def __init__(self, input_filters: int, output_filters: int, kernel_size: int, 
                 stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        self.input_filters = input_filters
        self.output_filters = output_filters
        
        # Expansion
        expanded_filters = input_filters * expand_ratio
        self.expand_conv = nn.Conv2d(input_filters, expanded_filters, 1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(expanded_filters) if expand_ratio != 1 else None
        
        # Depthwise
        self.depthwise_conv = nn.Conv2d(expanded_filters, expanded_filters, kernel_size, 
                                       stride, kernel_size//2, groups=expanded_filters, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_filters)
        
        # Squeeze and Excitation
        self.se = SqueezeExcitation(expanded_filters)
        
        # Pointwise
        self.pointwise_conv = nn.Conv2d(expanded_filters, output_filters, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(output_filters)
        
        self.activation = nn.SiLU(inplace=True)
        
    def forward(self, x):
        shortcut = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.activation(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        x = self.activation(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze and Excitation
        x = self.se(x)
        
        # Pointwise
        x = self.pointwise_bn(self.pointwise_conv(x))
        
        # Skip connection
        if (self.stride == 1 and self.input_filters == self.output_filters):
            x = x + shortcut
        
        return x

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation module"""
    
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = max(1, input_channels // squeeze_factor)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, squeeze_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, input_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

# === TRANSFORMER-BASED LANGUAGE MODELS ===

class BERTModel(nn.Module):
    """BERT model for text pattern recognition"""
    
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768, 
                 num_layers: int = 12, num_heads: int = 12, max_length: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        self.embeddings_norm = nn.LayerNorm(hidden_size)
        self.embeddings_dropout = nn.Dropout(0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BERTLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # Pooler
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embeddings_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings)
        
        # Transformer layers
        hidden_states = embeddings
        all_hidden_states = []
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
        
        # Pooler
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output,
            'hidden_states': all_hidden_states
        }

class BERTLayer(nn.Module):
    """BERT transformer layer"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = BERTAttention(hidden_size, num_heads)
        self.intermediate = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.layer_norm1(hidden_states + attention_output)
        
        # Feed forward
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.layer_norm2(attention_output + self.dropout(layer_output))
        
        return layer_output

class BERTAttention(nn.Module):
    """BERT multi-head attention"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Linear transformations
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = query_layer.view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)
        
        # Attention computation
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        attention_output = self.dense(context_layer)
        return attention_output

# === MULTIMODAL MODELS ===

class CLIP(nn.Module):
    """CLIP model for vision-language understanding"""
    
    def __init__(self, embed_dim: int = 512, image_resolution: int = 224, 
                 vision_layers: int = 12, text_layers: int = 12, vocab_size: int = 49408):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Vision encoder
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=32,
            width=768,
            layers=vision_layers,
            heads=12,
            output_dim=embed_dim
        )
        
        # Text encoder
        self.text_encoder = TextTransformer(
            vocab_size=vocab_size,
            embed_dim=512,
            layers=text_layers,
            heads=8,
            output_dim=embed_dim
        )
        
        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        return self.visual(image)
    
    def encode_text(self, text):
        return self.text_encoder(text)
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

class VisionTransformer(nn.Module):
    """Vision Transformer for CLIP"""
    
    def __init__(self, input_resolution: int, patch_size: int, width: int, 
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        
        self.transformer = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        
        return x

class TextTransformer(nn.Module):
    """Text Transformer for CLIP"""
    
    def __init__(self, vocab_size: int, embed_dim: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(77, embed_dim))
        
        self.transformer = nn.Sequential(*[ResidualAttentionBlock(embed_dim, heads) for _ in range(layers)])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.empty(embed_dim, output_dim))
        
    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # Take features from the eot embedding
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x

class ResidualAttentionBlock(nn.Module):
    """Residual attention block"""
    
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

# === REINFORCEMENT LEARNING ALGORITHMS ===

class DQN(nn.Module):
    """Deep Q-Network for pattern-based decision making"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.network(state)

class PPOActor(nn.Module):
    """PPO Actor network for policy-based learning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)

class PPOCritic(nn.Module):
    """PPO Critic network for value estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.network(state)

# === ALGORITHM REGISTRY FOR EDGE OF CHAOS TRAINING ===

class AlgorithmRegistry:
    """Registry of all implemented algorithms for systematic training"""
    
    def __init__(self):
        self.algorithms = {}
        self.pattern_extractors = {}
        self.chaos_metrics = defaultdict(list)
        
    def register_algorithm(self, name: str, algorithm_class, algorithm_type: str,
                          complexity_level: int, pattern_types: List[str]):
        """Register an algorithm with metadata"""
        self.algorithms[name] = {
            'class': algorithm_class,
            'type': algorithm_type,  # 'visual', 'audio', 'text', 'multimodal', 'rl'
            'complexity': complexity_level,  # 1-10
            'patterns': pattern_types,  # List of pattern types it extracts
            'chaos_threshold': 0.8 + (complexity_level * 0.02),  # Higher complexity = higher threshold
            'dream_cycles': max(3, complexity_level),  # More complex = more dream cycles needed
        }
    
    def get_algorithms_by_complexity(self, max_complexity: int = 10) -> Dict[str, Any]:
        """Get algorithms up to a certain complexity level"""
        return {name: info for name, info in self.algorithms.items() 
                if info['complexity'] <= max_complexity}
    
    def get_algorithms_by_type(self, algorithm_type: str) -> Dict[str, Any]:
        """Get algorithms of a specific type"""
        return {name: info for name, info in self.algorithms.items() 
                if info['type'] == algorithm_type}
    
    def create_training_sequence(self) -> List[Tuple[str, Dict]]:
        """Create optimal training sequence: simple -> complex"""
        sorted_algorithms = sorted(self.algorithms.items(), 
                                 key=lambda x: x[1]['complexity'])
        return sorted_algorithms

# Initialize and populate registry
algorithm_registry = AlgorithmRegistry()

# Register visual algorithms
algorithm_registry.register_algorithm('sobel_edges', None, 'visual', 1, ['edges', 'gradients'])
algorithm_registry.register_algorithm('harris_corners', None, 'visual', 2, ['corners', 'keypoints'])
algorithm_registry.register_algorithm('hough_lines', None, 'visual', 3, ['lines', 'geometry'])
algorithm_registry.register_algorithm('kmeans_clustering', None, 'visual', 4, ['clusters', 'grouping'])
algorithm_registry.register_algorithm('resnet50', ResNet50, 'visual', 7, ['features', 'hierarchy', 'classification'])
algorithm_registry.register_algorithm('efficientnet', EfficientNet, 'visual', 8, ['efficient_features', 'scaling'])
algorithm_registry.register_algorithm('swin_transformer', None, 'visual', 9, ['hierarchical_attention', 'windows'])

# Register text algorithms
algorithm_registry.register_algorithm('ngram', None, 'text', 2, ['sequences', 'prediction'])
algorithm_registry.register_algorithm('tfidf', None, 'text', 3, ['importance', 'frequency'])
algorithm_registry.register_algorithm('word2vec', None, 'text', 5, ['embeddings', 'semantics'])
algorithm_registry.register_algorithm('bert', BERTModel, 'text', 8, ['contextualized', 'bidirectional'])

# Register multimodal algorithms
algorithm_registry.register_algorithm('clip', CLIP, 'multimodal', 9, ['vision_language', 'alignment'])

# Register RL algorithms
algorithm_registry.register_algorithm('dqn', DQN, 'rl', 6, ['value_function', 'decision_making'])
algorithm_registry.register_algorithm('ppo', None, 'rl', 7, ['policy_gradient', 'optimization'])

def get_algorithm_training_plan(max_complexity: int = 10) -> Dict[str, Any]:
    """Get complete training plan respecting edge of chaos principles"""
    sequence = algorithm_registry.create_training_sequence()
    filtered_sequence = [(name, info) for name, info in sequence 
                        if info['complexity'] <= max_complexity]
    
    return {
        'training_sequence': filtered_sequence,
        'total_algorithms': len(filtered_sequence),
        'complexity_distribution': {
            f'level_{i}': len([info for _, info in filtered_sequence if info['complexity'] == i])
            for i in range(1, max_complexity + 1)
        }
    }
