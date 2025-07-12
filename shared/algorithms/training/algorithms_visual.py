# === COMPREHENSIVE ALGORITHM SUITE - MISSING IMPLEMENTATIONS ===
# Focus: Maximum pattern diversity for edge of chaos training
# Baby phase = establish many different connection types quickly
# Real learning happens in edge of chaos -> dream -> deep sleep -> dream -> wake cycles

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage, signal, optimize
from scipy.stats import entropy
import cv2
from typing import Tuple, List, Dict, Optional, Any
import math
import networkx as nx
from collections import defaultdict, Counter
import heapq

# === MISSING VISUAL ALGORITHMS ===

class AdvancedVisionAlgorithms:
    """Complete the missing visual processing algorithms for pattern diversity"""
    
    @staticmethod
    def hough_line_transform(image: np.ndarray, threshold: int = 100) -> List[Tuple[float, float]]:
        """ACTUAL Hough line detection - linear pattern recognition"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        
        line_params = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                line_params.append((rho, theta))
        
        return line_params
    
    @staticmethod
    def hough_circle_transform(image: np.ndarray, min_radius: int = 10, max_radius: int = 100) -> List[Tuple[int, int, int]]:
        """ACTUAL Hough circle detection - circular pattern recognition"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, 
                                  minRadius=min_radius, maxRadius=max_radius)
        
        circle_params = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                circle_params.append((x, y, r))
        
        return circle_params
    
    @staticmethod
    def watershed_segmentation(image: np.ndarray) -> np.ndarray:
        """ACTUAL watershed segmentation - region pattern recognition"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            markers = cv2.watershed(color_img, markers)
        
        return markers
    
    @staticmethod
    def kmeans_clustering(data: np.ndarray, k: int = 3, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """ACTUAL K-means clustering - grouping pattern recognition"""
        # Flatten image data if needed
        if len(data.shape) > 2:
            original_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
        else:
            original_shape = data.shape
            data = data.reshape(-1, 1)
        
        # Initialize centroids randomly
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels.reshape(original_shape[:-1] if len(original_shape) > 2 else original_shape), centroids
    
    @staticmethod
    def dbscan_clustering(data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """ACTUAL DBSCAN clustering - density pattern recognition"""
        if len(data.shape) > 2:
            data = data.reshape(-1, data.shape[-1])
        else:
            data = data.reshape(-1, 1)
        
        labels = np.full(data.shape[0], -1)
        cluster_id = 0
        
        for i in range(data.shape[0]):
            if labels[i] != -1:  # Already processed
                continue
            
            # Find neighbors
            distances = np.sqrt(np.sum((data - data[i])**2, axis=1))
            neighbors = np.where(distances <= eps)[0]
            
            if len(neighbors) < min_samples:
                labels[i] = -1  # Noise point
                continue
            
            # Start new cluster
            labels[i] = cluster_id
            seed_set = neighbors.tolist()
            
            j = 0
            while j < len(seed_set):
                current_point = seed_set[j]
                
                if labels[current_point] == -1:  # Change noise to border point
                    labels[current_point] = cluster_id
                elif labels[current_point] != -1:  # Already processed
                    j += 1
                    continue
                
                labels[current_point] = cluster_id
                
                # Find neighbors of current point
                distances = np.sqrt(np.sum((data - data[current_point])**2, axis=1))
                current_neighbors = np.where(distances <= eps)[0]
                
                if len(current_neighbors) >= min_samples:
                    seed_set.extend(current_neighbors.tolist())
                
                j += 1
            
            cluster_id += 1
        
        return labels

# === MISSING AUDIO ALGORITHMS ===

class AdvancedAudioAlgorithms:
    """Complete missing audio processing algorithms for pattern diversity"""
    
    @staticmethod
    def harmonic_analysis(signal: np.ndarray, sample_rate: int = 44100, 
                         num_harmonics: int = 10) -> Dict[str, np.ndarray]:
        """ACTUAL harmonic content analysis"""
        # Fundamental frequency detection
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Find fundamental frequency (highest peak in low frequency range)
        low_freq_mask = (freqs > 80) & (freqs < 1000)
        low_freq_fft = np.abs(fft[low_freq_mask])
        low_freqs = freqs[low_freq_mask]
        
        if len(low_freq_fft) == 0:
            return {'fundamental': 0, 'harmonics': np.zeros(num_harmonics), 'harmonic_freqs': np.zeros(num_harmonics)}
        
        fundamental_idx = np.argmax(low_freq_fft)
        fundamental_freq = low_freqs[fundamental_idx]
        
        # Extract harmonic content
        harmonics = []
        harmonic_freqs = []
        
        for h in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * h
            
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_amplitude = np.abs(fft[freq_idx])
            
            harmonics.append(harmonic_amplitude)
            harmonic_freqs.append(harmonic_freq)
        
        return {
            'fundamental': fundamental_freq,
            'harmonics': np.array(harmonics),
            'harmonic_freqs': np.array(harmonic_freqs),
            'harmonic_ratios': np.array(harmonics) / max(harmonics) if max(harmonics) > 0 else np.zeros(len(harmonics))
        }
    
    @staticmethod
    def onset_detection(signal: np.ndarray, sample_rate: int = 44100, 
                       hop_length: int = 512) -> np.ndarray:
        """ACTUAL onset detection - event timing patterns"""
        # Short-time energy
        frame_length = 2048
        energy = []
        
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        energy = np.array(energy)
        
        # Spectral flux
        stft = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            spectrum = np.abs(np.fft.fft(frame))
            stft.append(spectrum[:frame_length//2])
        
        stft = np.array(stft)
        spectral_flux = np.sum(np.diff(stft, axis=0), axis=1)
        spectral_flux = np.maximum(0, spectral_flux)  # Half-wave rectification
        
        # Combine energy and spectral flux
        onset_strength = spectral_flux[:-1] * energy[1:-1]
        
        # Peak picking
        threshold = np.mean(onset_strength) + 2 * np.std(onset_strength)
        onsets = []
        
        for i in range(1, len(onset_strength) - 1):
            if (onset_strength[i] > threshold and 
                onset_strength[i] > onset_strength[i-1] and 
                onset_strength[i] > onset_strength[i+1]):
                onset_time = i * hop_length / sample_rate
                onsets.append(onset_time)
        
        return np.array(onsets)
    
    @staticmethod
    def beat_tracking(signal: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
        """ACTUAL beat tracking - rhythmic pattern recognition"""
        # Get onset times
        onsets = AdvancedAudioAlgorithms.onset_detection(signal, sample_rate)
        
        if len(onsets) < 2:
            return {'tempo': 0, 'beats': np.array([]), 'intervals': np.array([])}
        
        # Calculate inter-onset intervals
        intervals = np.diff(onsets)
        
        # Estimate tempo using autocorrelation of onset times
        max_lag = int(sample_rate * 2)  # 2 seconds max
        onset_signal = np.zeros(len(signal))
        
        # Create onset signal
        for onset in onsets:
            onset_sample = int(onset * sample_rate)
            if onset_sample < len(onset_signal):
                onset_signal[onset_sample] = 1
        
        # Autocorrelation
        autocorr = np.correlate(onset_signal, onset_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find tempo (60-200 BPM range)
        min_period = int(sample_rate * 60 / 200)  # 200 BPM
        max_period = int(sample_rate * 60 / 60)   # 60 BPM
        
        if max_period < len(autocorr):
            tempo_range = autocorr[min_period:max_period]
            tempo_period = np.argmax(tempo_range) + min_period
            tempo = 60 * sample_rate / tempo_period
        else:
            tempo = 120  # Default
        
        # Generate beat times
        beat_interval = 60 / tempo
        beats = np.arange(0, len(signal) / sample_rate, beat_interval)
        
        return {
            'tempo': tempo,
            'beats': beats,
            'intervals': intervals,
            'onset_times': onsets
        }
    
    @staticmethod
    def pitch_detection(signal: np.ndarray, sample_rate: int = 44100) -> float:
        """ACTUAL pitch detection using autocorrelation"""
        if len(signal) < 1024:
            return 0.0
        
        # Apply window
        windowed = signal * np.hanning(len(signal))
        
        # Autocorrelation
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find pitch in reasonable range (80-1000 Hz)
        min_period = int(sample_rate / 1000)
        max_period = int(sample_rate / 80)
        
        if max_period < len(autocorr):
            pitch_range = autocorr[min_period:max_period]
            if len(pitch_range) > 0:
                pitch_period = np.argmax(pitch_range) + min_period
                return sample_rate / pitch_period
        
        return 0.0

# === MISSING TEXT ALGORITHMS ===

class AdvancedTextAlgorithms:
    """Complete missing text processing algorithms for pattern diversity"""
    
    @staticmethod
    def ngram_model(text: str, n: int = 3) -> Dict[Tuple[str, ...], Counter]:
        """ACTUAL n-gram language model"""
        words = text.lower().split()
        ngrams = defaultdict(Counter)
        
        # Generate n-grams
        for i in range(len(words) - n + 1):
            context = tuple(words[i:i+n-1])
            next_word = words[i+n-1]
            ngrams[context][next_word] += 1
        
        return dict(ngrams)
    
    @staticmethod
    def tfidf_vectorizer(documents: List[str], max_features: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """ACTUAL TF-IDF implementation"""
        # Tokenization and vocabulary building
        vocabulary = set()
        doc_words = []
        
        for doc in documents:
            words = doc.lower().split()
            doc_words.append(words)
            vocabulary.update(words)
        
        # Limit vocabulary size
        if len(vocabulary) > max_features:
            word_counts = Counter()
            for words in doc_words:
                word_counts.update(words)
            vocabulary = set([word for word, _ in word_counts.most_common(max_features)])
        
        vocabulary = sorted(list(vocabulary))
        vocab_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        
        # Calculate TF-IDF
        tfidf_matrix = np.zeros((len(documents), len(vocabulary)))
        
        # Document frequency
        df = np.zeros(len(vocabulary))
        for words in doc_words:
            unique_words = set(words) & set(vocabulary)
            for word in unique_words:
                df[vocab_to_idx[word]] += 1
        
        # Calculate TF-IDF for each document
        for doc_idx, words in enumerate(doc_words):
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in vocab_to_idx:
                    word_idx = vocab_to_idx[word]
                    tf = count / len(words)
                    idf = np.log(len(documents) / (df[word_idx] + 1))
                    tfidf_matrix[doc_idx, word_idx] = tf * idf
        
        return tfidf_matrix, vocabulary
    
    @staticmethod
    def word2vec_skipgram(sentences: List[List[str]], embedding_dim: int = 100, 
                         window_size: int = 5, epochs: int = 10) -> Dict[str, np.ndarray]:
        """ACTUAL Word2Vec Skip-gram implementation (simplified)"""
        # Build vocabulary
        vocabulary = set()
        for sentence in sentences:
            vocabulary.update(sentence)
        
        vocabulary = sorted(list(vocabulary))
        vocab_size = len(vocabulary)
        word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        
        # Initialize embeddings
        W1 = np.random.uniform(-1, 1, (vocab_size, embedding_dim))  # Input weights
        W2 = np.random.uniform(-1, 1, (embedding_dim, vocab_size))  # Output weights
        
        learning_rate = 0.01
        
        # Training
        for epoch in range(epochs):
            loss = 0
            
            for sentence in sentences:
                for center_idx, center_word in enumerate(sentence):
                    center_word_idx = word_to_idx[center_word]
                    
                    # Context words
                    start = max(0, center_idx - window_size)
                    end = min(len(sentence), center_idx + window_size + 1)
                    
                    for context_idx in range(start, end):
                        if context_idx != center_idx:
                            context_word = sentence[context_idx]
                            context_word_idx = word_to_idx[context_word]
                            
                            # Forward pass
                            h = W1[center_word_idx]
                            u = np.dot(h, W2)
                            y_pred = self._softmax(u)
                            
                            # Calculate loss
                            loss += -np.log(y_pred[context_word_idx])
                            
                            # Backward pass
                            e = y_pred.copy()
                            e[context_word_idx] -= 1
                            
                            # Update weights
                            dW2 = np.outer(h, e)
                            dW1 = np.dot(W2, e)
                            
                            W2 -= learning_rate * dW2
                            W1[center_word_idx] -= learning_rate * dW1
        
        # Return word embeddings
        embeddings = {}
        for word, idx in word_to_idx.items():
            embeddings[word] = W1[idx]
        
        return embeddings
    
    @staticmethod
    def _softmax(x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

# === NEURAL NETWORK ARCHITECTURES ===

class ResNetBlock(nn.Module):
    """Residual block for ResNet architectures"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet50(nn.Module):
    """ACTUAL ResNet-50 implementation for visual pattern recognition"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# === EDGE OF CHAOS TRAINING INTEGRATION ===

class EdgeOfChaosDetector:
    """
    Detect when model reaches edge of chaos during training
    Triggers training cool down: focus -> dream -> deep sleep -> dream -> wake
    """
    
    def __init__(self, chaos_threshold: float = 0.85, stability_threshold: float = 0.15):
        self.chaos_threshold = chaos_threshold
        self.stability_threshold = stability_threshold
        self.gradient_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.activation_history = deque(maxlen=50)
        
    def update_metrics(self, gradients: torch.Tensor, loss: float, activations: torch.Tensor):
        """Update chaos detection metrics"""
        # Gradient norm
        grad_norm = torch.norm(gradients).item()
        self.gradient_history.append(grad_norm)
        
        # Loss
        self.loss_history.append(loss)
        
        # Activation entropy (measure of chaos in activations)
        if activations.numel() > 0:
            flat_activations = activations.flatten()
            # Bin activations for entropy calculation
            hist, _ = np.histogram(flat_activations.cpu().detach().numpy(), bins=50, density=True)
            hist = hist + 1e-8  # Avoid log(0)
            activation_entropy = -np.sum(hist * np.log(hist))
            self.activation_history.append(activation_entropy)
    
    def detect_edge_of_chaos(self) -> Dict[str, Any]:
        """Detect if model has reached edge of chaos"""
        if len(self.gradient_history) < 10:
            return {'edge_detected': False, 'chaos_level': 0.0, 'stability_level': 1.0}
        
        # Calculate chaos indicators
        recent_grads = list(self.gradient_history)[-10:]
        recent_losses = list(self.loss_history)[-10:]
        recent_activations = list(self.activation_history)[-5:] if self.activation_history else [0]
        
        # Gradient instability
        grad_variance = np.var(recent_grads)
        grad_mean = np.mean(recent_grads)
        grad_instability = grad_variance / (grad_mean + 1e-8)
        
        # Loss oscillation
        loss_variance = np.var(recent_losses)
        loss_oscillation = loss_variance / (np.mean(recent_losses) + 1e-8)
        
        # Activation chaos
        activation_chaos = np.mean(recent_activations)
        
        # Combined chaos level
        chaos_level = np.mean([
            min(grad_instability / 10.0, 1.0),  # Normalize
            min(loss_oscillation / 5.0, 1.0),
            min(activation_chaos / 5.0, 1.0)
        ])
        
        edge_detected = chaos_level > self.chaos_threshold
        
        return {
            'edge_detected': edge_detected,
            'chaos_level': chaos_level,
            'grad_instability': grad_instability,
            'loss_oscillation': loss_oscillation,
            'activation_chaos': activation_chaos,
            'stability_level': 1.0 - chaos_level
        }

class TrainingCycleController:
    """
    Controls the training cycle: focus -> edge of chaos -> dream -> deep sleep -> dream -> wake
    Prevents pattern collapse by respecting chaos boundaries
    """
    
    def __init__(self):
        self.cycle_state = 'focus'  # focus, dream, deep_sleep, wake
        self.chaos_detector = EdgeOfChaosDetector()
        self.dream_cycles = 0
        self.rem_cycles_completed = 0
        self.target_rem_cycles = 3
        
    def update_training_state(self, gradients: torch.Tensor, loss: float, 
                            activations: torch.Tensor) -> Dict[str, Any]:
        """Update training state based on chaos detection"""
        # Update chaos metrics
        self.chaos_detector.update_metrics(gradients, loss, activations)
        chaos_info = self.chaos_detector.detect_edge_of_chaos()
        
        # State transitions
        if self.cycle_state == 'focus' and chaos_info['edge_detected']:
            # Transition to dream state
            self.cycle_state = 'dream'
            self.dream_cycles = 0
            return {
                'state_change': True,
                'new_state': 'dream',
                'action': 'start_dream_processing',
                'chaos_info': chaos_info
            }
        
        elif self.cycle_state == 'dream':
            self.dream_cycles += 1
            if self.dream_cycles >= 50:  # Dream processing cycles
                self.cycle_state = 'deep_sleep'
                return {
                    'state_change': True,
                    'new_state': 'deep_sleep',
                    'action': 'enter_deep_sleep',
                    'chaos_info': chaos_info
                }
        
        elif self.cycle_state == 'deep_sleep':
            # Non-learning deep sleep phase
            if self.rem_cycles_completed < self.target_rem_cycles:
                self.cycle_state = 'dream'
                self.rem_cycles_completed += 1
                self.dream_cycles = 0
                return {
                    'state_change': True,
                    'new_state': 'dream',
                    'action': 'return_to_dream',
                    'chaos_info': chaos_info
                }
            else:
                # Wake up - patterns should be processed
                self.cycle_state = 'wake'
                return {
                    'state_change': True,
                    'new_state': 'wake',
                    'action': 'wake_up',
                    'chaos_info': chaos_info
                }
        
        elif self.cycle_state == 'wake':
            # Check if ready to focus again
            if chaos_info['stability_level'] > 0.7:
                self.cycle_state = 'focus'
                self.rem_cycles_completed = 0
                return {
                    'state_change': True,
                    'new_state': 'focus',
                    'action': 'return_to_focus',
                    'chaos_info': chaos_info
                }
        
        return {
            'state_change': False,
            'current_state': self.cycle_state,
            'action': 'continue_current_state',
            'chaos_info': chaos_info
        }

# === PATTERN DIVERSITY METRICS ===

def calculate_pattern_diversity(activations_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate how diverse the learned patterns are across different algorithms"""
    diversity_metrics = {}
    
    for alg_name, activations in activations_dict.items():
        if activations.numel() == 0:
            diversity_metrics[alg_name] = 0.0
            continue
            
        # Flatten activations
        flat_acts = activations.flatten().cpu().detach().numpy()
        
        # Calculate diversity metrics
        # 1. Entropy (higher = more diverse)
        hist, _ = np.histogram(flat_acts, bins=50, density=True)
        hist = hist + 1e-8
        entropy_score = -np.sum(hist * np.log(hist))
        
        # 2. Variance (spread of activations)
        variance_score = np.var(flat_acts)
        
        # 3. Dynamic range
        range_score = np.max(flat_acts) - np.min(flat_acts)
        
        # Combined diversity score
        diversity_score = (entropy_score / 5.0 + variance_score / 10.0 + range_score / 2.0) / 3.0
        diversity_metrics[alg_name] = min(diversity_score, 1.0)
    
    return diversity_metrics