# === COMPLETE BABY BRAIN ALGORITHMS - 16 ALGORITHMS ===
# Location: shared/algorithms/baby/
# Purpose: Maximum connection diversity for early development
# Focus: Natural learning approaches with built-in success metrics

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from enum import Enum
import json

class LearningSuccess(Enum):
    EXCELLENT = "excellent"      # >90% success rate
    GOOD = "good"               # 70-90% success rate  
    MODERATE = "moderate"       # 50-70% success rate
    POOR = "poor"              # 30-50% success rate
    TERRIBLE = "terrible"      # <30% success rate

@dataclass
class BabyLearningMetric:
    """Track learning success for each baby brain algorithm"""
    algorithm_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    confidence_scores: List[float] = None
    connection_strength: float = 0.0
    pattern_diversity: float = 0.0
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = []
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    @property
    def learning_status(self) -> LearningSuccess:
        rate = self.success_rate
        if rate >= 0.9:
            return LearningSuccess.EXCELLENT
        elif rate >= 0.7:
            return LearningSuccess.GOOD
        elif rate >= 0.5:
            return LearningSuccess.MODERATE
        elif rate >= 0.3:
            return LearningSuccess.POOR
        else:
            return LearningSuccess.TERRIBLE

class BabyBrainTracker:
    """Central tracker for all baby brain learning metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, BabyLearningMetric] = {}
        self.cross_modal_connections: Dict[str, List[str]] = defaultdict(list)
        self.pattern_memory: Dict[str, List[Any]] = defaultdict(list)
        self.development_timeline: List[Dict[str, Any]] = []
        
    def record_learning_attempt(self, algorithm_name: str, success: bool, 
                              confidence: float = 0.5, connection_data: Dict = None):
        """Record a learning attempt and update metrics"""
        if algorithm_name not in self.metrics:
            self.metrics[algorithm_name] = BabyLearningMetric(algorithm_name)
        
        metric = self.metrics[algorithm_name]
        metric.total_attempts += 1
        if success:
            metric.successful_attempts += 1
        
        metric.confidence_scores.append(confidence)
        if len(metric.confidence_scores) > 20:
            metric.confidence_scores = metric.confidence_scores[-20:]
        
        # Update connection strength based on cross-modal data
        if connection_data:
            self._update_connections(algorithm_name, connection_data)
        
        # Log development milestone
        self.development_timeline.append({
            'algorithm': algorithm_name,
            'success': success,
            'confidence': confidence,
            'timestamp': time.time(),
            'total_attempts': metric.total_attempts
        })
    
    def _update_connections(self, algorithm_name: str, connection_data: Dict):
        """Update cross-modal connections"""
        for modality, strength in connection_data.items():
            if strength > 0.3:  # Threshold for meaningful connection
                self.cross_modal_connections[algorithm_name].append(modality)
    
    def get_development_report(self) -> Dict[str, Any]:
        """Generate comprehensive development report"""
        total_algorithms = len(self.metrics)
        successful_algorithms = sum(1 for m in self.metrics.values() if m.success_rate > 0.5)
        
        return {
            'algorithms_tried': total_algorithms,
            'successful_algorithms': successful_algorithms,
            'overall_success_rate': successful_algorithms / total_algorithms if total_algorithms > 0 else 0,
            'cross_modal_connections': len(self.cross_modal_connections),
            'development_milestones': len(self.development_timeline),
            'algorithm_performance': {name: metric.success_rate for name, metric in self.metrics.items()}
        }

# Global baby brain tracker
baby_tracker = BabyBrainTracker()

# === 1. CROSS-MODAL BABY LEARNING ===
class CrossModalBabyLearning:
    """
    Associate different sensory inputs like baby learning voice-face associations
    Essential for multi-sensory learning and pattern connection
    """
    
    def __init__(self, memory_size: int = 500):
        self.memory_size = memory_size
        self.associations: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        self.temporal_window = 2.0  # seconds
        self.recent_events: deque = deque(maxlen=30)
        
    def add_sensory_input(self, modality: str, data: Any, timestamp: float = None):
        """Add sensory input and find temporal associations"""
        if timestamp is None:
            timestamp = time.time()
            
        event = {
            'modality': modality,
            'data': data,
            'timestamp': timestamp,
            'features': self._extract_baby_features(modality, data)
        }
        
        self.recent_events.append(event)
        connections = self._find_temporal_associations(event)
        
        # Record learning attempt
        success = len(connections) > 0
        confidence = sum(c['strength'] for c in connections) / len(connections) if connections else 0.1
        baby_tracker.record_learning_attempt('cross_modal_learning', success, confidence, 
                                           {'connections': len(connections)})
        
        return event, connections
    
    def _extract_baby_features(self, modality: str, data: Any) -> Dict[str, float]:
        """Extract simple features babies would notice"""
        features = {}
        
        if modality == 'visual':
            if isinstance(data, np.ndarray):
                features['brightness'] = np.mean(data)
                features['color_variation'] = np.std(data)
                if len(data.shape) == 3:
                    features['red_avg'] = np.mean(data[:, :, 0])
                    features['green_avg'] = np.mean(data[:, :, 1])
                    features['blue_avg'] = np.mean(data[:, :, 2])
                
        elif modality == 'auditory':
            if isinstance(data, np.ndarray):
                features['volume'] = np.mean(np.abs(data))
                features['pitch_estimate'] = self._simple_pitch_detection(data)
                features['rhythm_pattern'] = self._simple_rhythm_detection(data)
                
        elif modality == 'tactile':
            if isinstance(data, dict):
                features['pressure'] = data.get('pressure', 0.0)
                features['temperature'] = data.get('temperature', 0.5)
                features['texture'] = data.get('texture', 0.5)
                
        return features
    
    def _simple_pitch_detection(self, audio: np.ndarray) -> float:
        """Very simple pitch detection for baby brain"""
        if len(audio) < 100:
            return 0.0
        # Simple zero-crossing rate based pitch estimation
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        if len(zero_crossings) > 1:
            return len(zero_crossings) / len(audio) * 1000  # Normalized
        return 0.0
    
    def _simple_rhythm_detection(self, audio: np.ndarray) -> float:
        """Simple rhythm pattern detection"""
        if len(audio) < 1000:
            return 0.0
        # Simple energy variation detection
        energy = np.abs(audio)
        energy_diff = np.diff(energy)
        peaks = np.where(energy_diff > np.std(energy_diff))[0]
        return len(peaks) / len(audio) if len(audio) > 0 else 0.0
    
    def _find_temporal_associations(self, new_event: Dict) -> List[Dict]:
        """Find temporal associations between events"""
        associations = []
        current_time = new_event['timestamp']
        
        for old_event in self.recent_events:
            if old_event == new_event:
                continue
                
            time_diff = abs(current_time - old_event['timestamp'])
            if time_diff <= self.temporal_window:
                # Calculate association strength
                feature_similarity = self._calculate_feature_similarity(
                    old_event['features'], new_event['features']
                )
                temporal_strength = 1.0 - (time_diff / self.temporal_window)
                
                association_strength = feature_similarity * temporal_strength
                
                if association_strength > 0.2:  # Minimum threshold
                    associations.append({
                        'from_modality': old_event['modality'],
                        'to_modality': new_event['modality'],
                        'strength': association_strength,
                        'time_gap': time_diff
                    })
        
        return associations
    
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between feature sets"""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        similarities = []
        for feature in common_features:
            val1, val2 = features1[feature], features2[feature]
            if abs(val1) + abs(val2) == 0:
                similarities.append(1.0)
            else:
                max_val = max(abs(val1), abs(val2), 0.1)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(similarity)
        
        return np.mean(similarities)

# === 2. NURSERY PATTERN MEMORY ===
class NurseryPatternMemory:
    """
    Simple pattern storage for basic learning: colors, shapes, sounds, numbers
    Essential for fundamental categorization and recognition
    """
    
    def __init__(self):
        self.patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.pattern_confidence: Dict[str, float] = {}
        self.recognition_history: List[Dict] = []
        
    def learn_pattern(self, category: str, pattern_data: Any, label: str = None) -> str:
        """Learn a simple pattern"""
        pattern_id = f"{category}_{len(self.patterns[category])}"
        
        pattern = {
            'id': pattern_id,
            'category': category,
            'data': pattern_data,
            'label': label,
            'features': self._extract_pattern_features(category, pattern_data),
            'learned_time': time.time(),
            'recognition_count': 0
        }
        
        self.patterns[category].append(pattern)
        self.pattern_confidence[pattern_id] = 0.5  # Start with medium confidence
        
        # Record learning
        baby_tracker.record_learning_attempt('nursery_pattern_memory', True, 0.5,
                                           {'category': category, 'total_patterns': len(self.patterns[category])})
        
        return pattern_id
    
    def recognize_pattern(self, category: str, test_data: Any) -> Dict[str, Any]:
        """Recognize a pattern from memory"""
        if category not in self.patterns:
            return {'recognized': False, 'confidence': 0.0}
        
        test_features = self._extract_pattern_features(category, test_data)
        best_match = None
        best_similarity = 0.0
        
        for pattern in self.patterns[category]:
            similarity = self._calculate_pattern_similarity(test_features, pattern['features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
        
        # Recognition threshold
        recognized = best_similarity > 0.6
        if recognized and best_match:
            best_match['recognition_count'] += 1
            self.pattern_confidence[best_match['id']] = min(
                self.pattern_confidence[best_match['id']] + 0.1, 1.0
            )
        
        result = {
            'recognized': recognized,
            'confidence': best_similarity,
            'pattern_id': best_match['id'] if best_match else None,
            'label': best_match['label'] if best_match else None
        }
        
        # Record recognition attempt
        baby_tracker.record_learning_attempt('nursery_pattern_memory', recognized, best_similarity)
        
        return result
    
    def _extract_pattern_features(self, category: str, data: Any) -> Dict[str, float]:
        """Extract features specific to pattern category"""
        features = {}
        
        if category == 'color':
            if isinstance(data, (list, tuple, np.ndarray)) and len(data) >= 3:
                features['red'] = float(data[0])
                features['green'] = float(data[1])
                features['blue'] = float(data[2])
                features['brightness'] = (features['red'] + features['green'] + features['blue']) / 3
                features['saturation'] = max(features['red'], features['green'], features['blue']) - \
                                       min(features['red'], features['green'], features['blue'])
                
        elif category == 'shape':
            if isinstance(data, np.ndarray):
                # Simple shape features
                features['area'] = np.sum(data > 0) if data.dtype == bool else np.sum(data)
                
                # Find contours for more shape features
                if data.dtype != np.uint8:
                    data = (data * 255).astype(np.uint8)
                contours, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    features['perimeter'] = cv2.arcLength(largest_contour, True)
                    features['aspect_ratio'] = self._calculate_aspect_ratio(largest_contour)
                    features['circularity'] = self._calculate_circularity(largest_contour)
                
        elif category == 'sound':
            if isinstance(data, np.ndarray):
                features['energy'] = np.sum(data ** 2)
                features['zero_crossings'] = len(np.where(np.diff(np.sign(data)))[0])
                features['peak_frequency'] = self._find_peak_frequency(data)
                
        elif category == 'number':
            if isinstance(data, (int, float)):
                features['value'] = float(data)
                features['magnitude'] = abs(float(data))
                features['is_positive'] = 1.0 if data >= 0 else 0.0
                features['digit_count'] = len(str(abs(int(data))))
        
        return features
    
    def _calculate_aspect_ratio(self, contour) -> float:
        """Calculate aspect ratio of contour"""
        x, y, w, h = cv2.boundingRect(contour)
        return w / h if h > 0 else 1.0
    
    def _calculate_circularity(self, contour) -> float:
        """Calculate circularity of contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)
        return 0.0
    
    def _find_peak_frequency(self, audio: np.ndarray) -> float:
        """Find peak frequency in audio signal"""
        if len(audio) < 2:
            return 0.0
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio))
        peak_idx = np.argmax(np.abs(fft))
        return abs(freqs[peak_idx])
    
    def _calculate_pattern_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between pattern features"""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        similarities = []
        for feature in common_features:
            val1, val2 = features1[feature], features2[feature]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                max_val = max(abs(val1), abs(val2), 0.01)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0, similarity))
        
        return np.mean(similarities)

# === 3. BLUR TOLERANCE PROCESSING ===
class BlurToleranceProcessing:
    """
    Process blurry images like baby vision - essential for early visual development
    Babies start with poor visual acuity and gradually improve
    """
    
    def __init__(self):
        self.blur_levels = [5, 10, 15, 20, 25]  # Different blur intensities
        self.recognition_threshold = 0.4
        
    def process_blurry_image(self, image: np.ndarray, blur_level: int = 15) -> Dict[str, Any]:
        """Process image with blur tolerance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Apply blur to simulate baby vision
        blurred = cv2.GaussianBlur(gray, (blur_level, blur_level), 0)
        
        # Extract basic features that work with blur
        features = self._extract_blur_tolerant_features(blurred)
        
        # Simple pattern recognition on blurred image
        recognition_result = self._recognize_basic_patterns(blurred)
        
        success = recognition_result['confidence'] > self.recognition_threshold
        baby_tracker.record_learning_attempt('blur_tolerance', success, recognition_result['confidence'],
                                           {'blur_level': blur_level})
        
        return {
            'original_shape': image.shape,
            'blur_level': blur_level,
            'features': features,
            'recognition': recognition_result,
            'success': success
        }
    
    def _extract_blur_tolerant_features(self, blurred_image: np.ndarray) -> Dict[str, float]:
        """Extract features that work well with blurred images"""
        features = {}
        
        # Global intensity features
        features['mean_intensity'] = np.mean(blurred_image)
        features['intensity_std'] = np.std(blurred_image)
        
        # Large-scale contrast features
        features['global_contrast'] = np.std(blurred_image) / (np.mean(blurred_image) + 1)
        
        # Basic shape detection using low-frequency components
        # Downsample for basic shape analysis
        small = cv2.resize(blurred_image, (32, 32))
        features['top_heavy'] = np.mean(small[:16, :]) / (np.mean(small[16:, :]) + 1)
        features['left_heavy'] = np.mean(small[:, :16]) / (np.mean(small[:, 16:]) + 1)
        
        # Large edge detection (works better with blur)
        edges = cv2.Canny(blurred_image, 30, 70)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def _recognize_basic_patterns(self, blurred_image: np.ndarray) -> Dict[str, Any]:
        """Recognize basic patterns in blurred image"""
        # Very simple pattern recognition suitable for blurred images
        h, w = blurred_image.shape
        
        # Check for face-like patterns (dark regions where eyes might be)
        top_third = blurred_image[:h//3, :]
        middle_third = blurred_image[h//3:2*h//3, :]
        
        # Simple heuristics for face-like pattern
        face_score = 0.0
        if np.mean(middle_third) < np.mean(top_third):  # Eyes darker than forehead
            face_score += 0.3
        
        # Check for roughly symmetric dark regions in eye area
        left_eye_region = middle_third[:, :w//3]
        right_eye_region = middle_third[:, 2*w//3:]
        if abs(np.mean(left_eye_region) - np.mean(right_eye_region)) < 20:  # Similar intensity
            face_score += 0.2
        
        # Check for circular/oval shape
        contours, _ = cv2.findContours(cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)[1], 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:  # Roughly circular
                    face_score += 0.3
        
        # Additional pattern checks
        object_score = min(face_score + 0.2, 1.0)  # Objects slightly easier to detect
        
        return {
            'face_likelihood': face_score,
            'object_likelihood': object_score,
            'confidence': max(face_score, object_score),
            'pattern_type': 'face' if face_score > object_score else 'object'
        }

# === 4. VOICE FAMILIARITY LEARNING ===
class VoiceFamiliarityLearning:
    """
    Learn to recognize familiar voices (mama, dada) through repetition
    Essential for bonding and security
    """
    
    def __init__(self):
        self.voice_profiles: Dict[str, Dict] = {}
        self.recognition_threshold = 0.6
        
    def learn_voice(self, voice_id: str, audio_sample: np.ndarray, label: str = None) -> Dict[str, Any]:
        """Learn a voice profile from audio sample"""
        if len(audio_sample) < 1000:  # Minimum sample length
            return {'success': False, 'reason': 'Sample too short'}
        
        # Extract voice features
        features = self._extract_voice_features(audio_sample)
        
        if voice_id not in self.voice_profiles:
            self.voice_profiles[voice_id] = {
                'label': label or voice_id,
                'features': [],
                'learning_count': 0,
                'recognition_count': 0,
                'last_heard': time.time()
            }
        
        # Add features to profile
        self.voice_profiles[voice_id]['features'].append(features)
        self.voice_profiles[voice_id]['learning_count'] += 1
        self.voice_profiles[voice_id]['last_heard'] = time.time()
        
        # Keep only recent features (sliding window)
        if len(self.voice_profiles[voice_id]['features']) > 10:
            self.voice_profiles[voice_id]['features'] = self.voice_profiles[voice_id]['features'][-10:]
        
        success = True
        confidence = min(0.3 + (self.voice_profiles[voice_id]['learning_count'] * 0.1), 1.0)
        baby_tracker.record_learning_attempt('voice_familiarity', success, confidence,
                                           {'voice_id': voice_id, 'samples': len(self.voice_profiles[voice_id]['features'])})
        
        return {
            'success': True,
            'voice_id': voice_id,
            'samples_learned': len(self.voice_profiles[voice_id]['features']),
            'confidence': confidence
        }
    
    def recognize_voice(self, audio_sample: np.ndarray) -> Dict[str, Any]:
        """Recognize a voice from learned profiles"""
        if len(audio_sample) < 1000:
            return {'recognized': False, 'reason': 'Sample too short'}
        
        test_features = self._extract_voice_features(audio_sample)
        
        best_match = None
        best_similarity = 0.0
        
        for voice_id, profile in self.voice_profiles.items():
            if not profile['features']:
                continue
                
            # Calculate average similarity to all stored samples
            similarities = []
            for stored_features in profile['features']:
                similarity = self._calculate_voice_similarity(test_features, stored_features)
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = voice_id
        
        recognized = best_similarity > self.recognition_threshold
        if recognized and best_match:
            self.voice_profiles[best_match]['recognition_count'] += 1
            self.voice_profiles[best_match]['last_heard'] = time.time()
        
        baby_tracker.record_learning_attempt('voice_familiarity', recognized, best_similarity)
        
        return {
            'recognized': recognized,
            'voice_id': best_match,
            'confidence': best_similarity,
            'label': self.voice_profiles[best_match]['label'] if best_match else None
        }
    
    def _extract_voice_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract voice characteristics"""
        features = {}
        
        # Basic audio properties
        features['energy'] = np.mean(audio ** 2)
        features['zero_crossing_rate'] = len(np.where(np.diff(np.sign(audio)))[0]) / len(audio)
        
        # Simple pitch estimation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 100:
            peak_idx = np.argmax(autocorr[20:100]) + 20  # Avoid zero lag
            features['pitch_estimate'] = peak_idx
        else:
            features['pitch_estimate'] = 50  # Default
        
        # Spectral features (simplified)
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio))
        magnitude = np.abs(fft)
        
        # Find dominant frequencies
        peak_indices = np.argsort(magnitude)[-5:]  # Top 5 peaks
        features['dominant_freq_1'] = abs(freqs[peak_indices[-1]]) if len(peak_indices) > 0 else 0
        features['dominant_freq_2'] = abs(freqs[peak_indices[-2]]) if len(peak_indices) > 1 else 0
        
        # Spectral centroid (brightness)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        features['spectral_centroid'] = abs(spectral_centroid)
        
        # Roughness/smoothness
        features['spectral_rolloff'] = self._calculate_spectral_rolloff(freqs, magnitude)
        
        return features
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, magnitude: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff point"""
        total_energy = np.sum(magnitude)
        cumulative_energy = np.cumsum(magnitude)
        rolloff_point = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
        if len(rolloff_point) > 0:
            return abs(freqs[rolloff_point[0]])
        return abs(freqs[-1])
    
    def _calculate_voice_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between voice features"""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
        
        similarities = []
        for feature in common_features:
            val1, val2 = features1[feature], features2[feature]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                # Normalize by the larger value
                max_val = max(abs(val1), abs(val2), 0.001)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0, similarity))
        
        return np.mean(similarities)

# === 5. COLOR-SHAPE ASSOCIATION ===
class ColorShapeAssociation:
    """
    Learn basic associations between colors and shapes
    Essential for object recognition and categorization
    """
    
    def __init__(self):
        self.associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.learning_history: List[Dict] = []
        
    def learn_association(self, color: Tuple[float, float, float], shape: str, 
                         confidence: float = 1.0) -> Dict[str, Any]:
        """Learn a color-shape association"""
        color_name = self._classify_color(color)
        
        # Update association strength
        self.associations[color_name][shape] += confidence * 0.1
        self.associations[color_name][shape] = min(self.associations[color_name][shape], 1.0)
        
        # Record learning
        learning_record = {
            'color': color_name,
            'shape': shape,
            'confidence': confidence,
            'timestamp': time.time(),
            'association_strength': self.associations[color_name][shape]
        }
        self.learning_history.append(learning_record)
        
        success = True
        baby_tracker.record_learning_attempt('color_shape_association', success, confidence,
                                           {'color': color_name, 'shape': shape})
        
        return {
            'success': True,
            'color_name': color_name,
            'shape': shape,
            'association_strength': self.associations[color_name][shape]
        }
    
    def predict_shape_from_color(self, color: Tuple[float, float, float]) -> Dict[str, Any]:
        """Predict most likely shape based on color"""
        color_name = self._classify_color(color)
        
        if color_name not in self.associations:
            return {'predicted': False, 'reason': 'Color not learned'}
        
        # Find shape with highest association
        best_shape = max(self.associations[color_name].items(), 
                        key=lambda x: x[1], default=(None, 0))
        
        predicted = best_shape[1] > 0.3  # Threshold for prediction
        confidence = best_shape[1]
        
        baby_tracker.record_learning_attempt('color_shape_association', predicted, confidence)
        
        return {
            'predicted': predicted,
            'shape': best_shape[0],
            'confidence': confidence,
            'all_associations': dict(self.associations[color_name])
        }
    
    def _classify_color(self, color: Tuple[float, float, float]) -> str:
        """Classify color into basic categories babies learn"""
        r, g, b = color
        
        # Normalize to 0-1 range if needed
        if max(r, g, b) > 1:
            r, g, b = r/255, g/255, b/255
        
        # Simple color classification
        if r > 0.7 and g < 0.3 and b < 0.3:
            return 'red'
        elif g > 0.7 and r < 0.3 and b < 0.3:
            return 'green'
        elif b > 0.7 and r < 0.3 and g < 0.3:
            return 'blue'
        elif r > 0.8 and g > 0.8 and b < 0.3:
            return 'yellow'
        elif r > 0.6 and g < 0.4 and b > 0.6:
            return 'purple'
        elif r > 0.8 and g > 0.5 and b < 0.3:
            return 'orange'
        elif r > 0.8 and g > 0.8 and b > 0.8:
            return 'white'
        elif r < 0.2 and g < 0.2 and b < 0.2:
            return 'black'
        else:
            return 'mixed'

# === 6. MOVEMENT TRACKING ===
class MovementTracking:
    """
    Track moving objects in visual field - essential for visual development
    Babies are naturally drawn to movement
    """
    
    def __init__(self):
        self.tracked_objects: List[Dict] = []
        self.movement_history: List[Dict] = []
        self.frame_buffer: deque = deque(maxlen=5)
        
    def track_movement(self, current_frame: np.ndarray) -> Dict[str, Any]:
        """Track movement between frames"""
        if len(current_frame.shape) == 3:
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_current = current_frame.astype(np.uint8)
        
        self.frame_buffer.append(gray_current)
        
        if len(self.frame_buffer) < 2:
            return {'movements_detected': 0, 'success': False}
        
        # Calculate frame difference
        prev_frame = self.frame_buffer[-2]
        frame_diff = cv2.absdiff(gray_current, prev_frame)
        
        # Threshold to find moving regions
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        movements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate movement properties
                movement = {
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2),
                    'timestamp': time.time()
                }
                movements.append(movement)
        
        # Track object persistence
        tracked_movements = self._track_object_persistence(movements)
        
        success = len(tracked_movements) > 0
        confidence = min(len(tracked_movements) * 0.3, 1.0)
        
        baby_tracker.record_learning_attempt('movement_tracking', success, confidence,
                                           {'movements': len(tracked_movements)})
        
        return {
            'movements_detected': len(tracked_movements),
            'movements': tracked_movements,
            'success': success,
            'confidence': confidence
        }
    
    def _track_object_persistence(self, current_movements: List[Dict]) -> List[Dict]:
        """Track objects across frames for persistence"""
        tracked = []
        
        for movement in current_movements:
            # Find if this movement matches a previous tracked object
            best_match = None
            best_distance = float('inf')
            
            for tracked_obj in self.tracked_objects:
                if time.time() - tracked_obj['last_seen'] < 2.0:  # Recent objects only
                    distance = np.sqrt((movement['center'][0] - tracked_obj['center'][0])**2 + 
                                     (movement['center'][1] - tracked_obj['center'][1])**2)
                    if distance < best_distance and distance < 50:  # Maximum movement threshold
                        best_distance = distance
                        best_match = tracked_obj
            
            if best_match:
                # Update existing tracked object
                best_match['center'] = movement['center']
                best_match['bbox'] = movement['bbox']
                best_match['last_seen'] = time.time()
                best_match['track_count'] += 1
                tracked.append(best_match)
            else:
                # New tracked object
                new_tracked = {
                    'id': len(self.tracked_objects),
                    'center': movement['center'],
                    'bbox': movement['bbox'],
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'track_count': 1
                }
                self.tracked_objects.append(new_tracked)
                tracked.append(new_tracked)
        
        return tracked

# === 7. FACE DETECTION SIMPLE ===
class FaceDetectionSimple:
    """
    Basic face-like pattern detection for baby brain
    Babies are naturally drawn to faces and face-like patterns
    """
    
    def __init__(self):
        self.face_templates = self._create_simple_face_templates()
        self.detection_threshold = 0.4
        
    def detect_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect face-like patterns in image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Resize to standard size for template matching
        resized = cv2.resize(gray, (64, 64))
        
        face_scores = []
        for template in self.face_templates:
            score = self._match_face_template(resized, template)
            face_scores.append(score)
        
        max_score = max(face_scores) if face_scores else 0.0
        face_detected = max_score > self.detection_threshold
        
        # Additional heuristic checks
        heuristic_score = self._heuristic_face_check(resized)
        combined_score = (max_score + heuristic_score) / 2
        
        baby_tracker.record_learning_attempt('face_detection_simple', face_detected, combined_score)
        
        return {
            'face_detected': face_detected,
            'confidence': combined_score,
            'template_scores': face_scores,
            'heuristic_score': heuristic_score
        }
    
    def _create_simple_face_templates(self) -> List[np.ndarray]:
        """Create simple face templates for matching"""
        templates = []
        
        # Template 1: Basic oval with two dark spots (eyes)
        template1 = np.ones((32, 24), dtype=np.uint8) * 200  # Light background
        # Eyes
        template1[8:12, 6:10] = 50   # Left eye
        template1[8:12, 14:18] = 50  # Right eye
        # Mouth
        template1[20:24, 10:14] = 100  # Mouth area
        templates.append(template1)
        
        # Template 2: Round face
        template2 = np.ones((32, 32), dtype=np.uint8) * 200
        # Create circular face outline
        center = (16, 16)
        cv2.circle(template2, center, 14, 150, -1)  # Face oval
        # Eyes
        cv2.circle(template2, (11, 12), 2, 50, -1)  # Left eye
        cv2.circle(template2, (21, 12), 2, 50, -1)  # Right eye
        # Mouth
        cv2.ellipse(template2, (16, 22), (4, 2), 0, 0, 180, 100, -1)
        templates.append(template2)
        
        return templates
    
    def _match_face_template(self, image: np.ndarray, template: np.ndarray) -> float:
        """Match image against face template"""
        # Resize image to match template
        h, w = template.shape
        resized_image = cv2.resize(image, (w, h))
        
        # Normalize both images
        norm_image = cv2.equalizeHist(resized_image)
        norm_template = cv2.equalizeHist(template)
        
        # Calculate normalized cross correlation
        result = cv2.matchTemplate(norm_image, norm_template, cv2.TM_CCOEFF_NORMED)
        return float(result[0, 0])
    
    def _heuristic_face_check(self, image: np.ndarray) -> float:
        """Use simple heuristics to check for face-like features"""
        h, w = image.shape
        score = 0.0
        
        # Check for symmetry (faces are roughly symmetric)
        left_half = image[:, :w//2]
        right_half = np.fliplr(image[:, w//2:])
        if right_half.shape[1] == left_half.shape[1]:
            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
            score += symmetry * 0.3
        
        # Check for dark regions in eye area (upper third)
        eye_region = image[:h//3, :]
        middle_region = image[h//3:2*h//3, :]
        
        if np.mean(eye_region) < np.mean(middle_region):
            score += 0.2  # Eyes darker than face
        
        # Check for oval/circular shape
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:  # Reasonably circular
                    score += 0.3
        
        return min(score, 1.0)

# === 8. EMOTIONAL TONE DETECTION ===
class EmotionalToneDetection:
    """
    Recognize emotional content in voice - essential for understanding caregiver states
    Babies are very sensitive to emotional tone
    """
    
    def __init__(self):
        self.emotion_profiles = {
            'happy': {'pitch_high': True, 'energy_high': True, 'variation_high': True},
            'sad': {'pitch_low': True, 'energy_low': True, 'variation_low': True},
            'angry': {'pitch_variable': True, 'energy_high': True, 'variation_high': True},
            'calm': {'pitch_stable': True, 'energy_medium': True, 'variation_low': True}
        }
        
    def detect_emotion(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect emotional tone in audio"""
        if len(audio) < 1000:
            return {'emotion': 'unknown', 'confidence': 0.0}
        
        # Extract emotional features
        features = self._extract_emotional_features(audio)
        
        # Calculate emotion scores
        emotion_scores = {}
        for emotion, profile in self.emotion_profiles.items():
            score = self._calculate_emotion_score(features, profile)
            emotion_scores[emotion] = score
        
        # Find best match
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        detected_emotion = best_emotion[0]
        confidence = best_emotion[1]
        
        success = confidence > 0.3
        baby_tracker.record_learning_attempt('emotional_tone_detection', success, confidence,
                                           {'emotion': detected_emotion})
        
        return {
            'emotion': detected_emotion,
            'confidence': confidence,
            'all_scores': emotion_scores,
            'features': features
        }
    
    def _extract_emotional_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features related to emotional content"""
        features = {}
        
        # Energy/amplitude features
        features['mean_energy'] = np.mean(audio ** 2)
        features['energy_variation'] = np.std(audio ** 2)
        
        # Pitch estimation through autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 100:
            # Find pitch peaks
            peaks = []
            for i in range(20, min(200, len(autocorr))):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append((i, autocorr[i]))
            
            if peaks:
                # Primary pitch
                primary_peak = max(peaks, key=lambda x: x[1])
                features['primary_pitch'] = primary_peak[0]
                
                # Pitch variation
                pitch_values = [p[0] for p in peaks[:5]]  # Top 5 peaks
                features['pitch_variation'] = np.std(pitch_values) if len(pitch_values) > 1 else 0
            else:
                features['primary_pitch'] = 50  # Default
                features['pitch_variation'] = 0
        
        # Tempo/rhythm features
        # Simple onset detection
        diff_audio = np.diff(audio)
        onset_strength = np.maximum(0, diff_audio)  # Half-wave rectification
        
        # Find tempo through periodicity
        onset_autocorr = np.correlate(onset_strength, onset_strength, mode='full')
        onset_autocorr = onset_autocorr[len(onset_autocorr)//2:]
        
        if len(onset_autocorr) > 100:
            tempo_peak = np.argmax(onset_autocorr[10:100]) + 10
            features['tempo_estimate'] = tempo_peak
        else:
            features['tempo_estimate'] = 50
        
        # Spectral features
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio))
        
        # Spectral centroid (brightness)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        features['spectral_brightness'] = abs(spectral_centroid)
        
        return features
    
    def _calculate_emotion_score(self, features: Dict[str, float], profile: Dict[str, bool]) -> float:
        """Calculate how well features match emotion profile"""
        score = 0.0
        matches = 0
        
        # Check energy level
        if 'energy_high' in profile and profile['energy_high']:
            if features['mean_energy'] > 0.1:  # High energy threshold
                score += 1.0
            matches += 1
        elif 'energy_low' in profile and profile['energy_low']:
            if features['mean_energy'] < 0.05:  # Low energy threshold
                score += 1.0
            matches += 1
        elif 'energy_medium' in profile and profile['energy_medium']:
            if 0.05 <= features['mean_energy'] <= 0.1:  # Medium energy
                score += 1.0
            matches += 1
        
        # Check pitch characteristics
        if 'pitch_high' in profile and profile['pitch_high']:
            if features['primary_pitch'] > 80:  # High pitch
                score += 1.0
            matches += 1
        elif 'pitch_low' in profile and profile['pitch_low']:
            if features['primary_pitch'] < 40:  # Low pitch
                score += 1.0
            matches += 1
        elif 'pitch_stable' in profile and profile['pitch_stable']:
            if features['pitch_variation'] < 10:  # Stable pitch
                score += 1.0
            matches += 1
        elif 'pitch_variable' in profile and profile['pitch_variable']:
            if features['pitch_variation'] > 20:  # Variable pitch
                score += 1.0
            matches += 1
        
        # Check variation
        if 'variation_high' in profile and profile['variation_high']:
            if features['energy_variation'] > 0.02:
                score += 1.0
            matches += 1
        elif 'variation_low' in profile and profile['variation_low']:
            if features['energy_variation'] < 0.01:
                score += 1.0
            matches += 1
        
        return score / matches if matches > 0 else 0.0

# === 9-16. REMAINING BABY BRAIN ALGORITHMS ===

class ObjectPermanence:
    """Understanding that objects exist when not visible - key cognitive milestone"""
    
    def __init__(self):
        self.tracked_objects = {}
        self.permanence_threshold = 3.0  # seconds
        
    def track_object_appearance(self, object_id: str, visible: bool, timestamp: float = None):
        """Track when objects appear and disappear"""
        if timestamp is None:
            timestamp = time.time()
        
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'visible': visible,
                'disappearance_count': 0,
                'reappearance_count': 0
            }
        
        obj = self.tracked_objects[object_id]
        
        if visible and not obj['visible']:
            # Object reappeared
            obj['reappearance_count'] += 1
            obj['visible'] = True
            obj['last_seen'] = timestamp
        elif not visible and obj['visible']:
            # Object disappeared
            obj['disappearance_count'] += 1
            obj['visible'] = False
        
        if visible:
            obj['last_seen'] = timestamp
        
        # Test permanence understanding
        permanence_score = obj['reappearance_count'] / max(obj['disappearance_count'], 1)
        success = permanence_score > 0.5
        
        baby_tracker.record_learning_attempt('object_permanence', success, permanence_score)
        
        return {
            'object_id': object_id,
            'permanence_score': permanence_score,
            'understands_permanence': success
        }

class CauseEffectSimple:
    """Basic understanding of cause and effect relationships"""
    
    def __init__(self):
        self.cause_effect_pairs = []
        self.learning_threshold = 0.6
    
    def learn_cause_effect(self, cause: str, effect: str, delay: float = 0.5):
        """Learn a cause-effect relationship"""
        pair = {
            'cause': cause,
            'effect': effect,
            'delay': delay,
            'occurrences': 1,
            'timestamp': time.time()
        }
        
        # Check if this pair already exists
        existing = None
        for p in self.cause_effect_pairs:
            if p['cause'] == cause and p['effect'] == effect:
                existing = p
                break
        
        if existing:
            existing['occurrences'] += 1
            existing['timestamp'] = time.time()
            confidence = min(existing['occurrences'] * 0.2, 1.0)
        else:
            self.cause_effect_pairs.append(pair)
            confidence = 0.2
        
        success = confidence > self.learning_threshold
        baby_tracker.record_learning_attempt('cause_effect_simple', success, confidence)
        
        return {'learned': success, 'confidence': confidence}

class TemporalSequenceBasic:
    """Learning simple temporal patterns and sequences"""
    
    def __init__(self):
        self.sequences = []
        self.current_sequence = []
        self.sequence_window = 5
    
    def add_event(self, event: str, timestamp: float = None):
        """Add event to current sequence"""
        if timestamp is None:
            timestamp = time.time()
        
        self.current_sequence.append({'event': event, 'timestamp': timestamp})
        
        if len(self.current_sequence) > self.sequence_window:
            self.current_sequence = self.current_sequence[-self.sequence_window:]
        
        # Check for pattern recognition
        pattern_score = self._recognize_pattern()
        success = pattern_score > 0.3
        
        baby_tracker.record_learning_attempt('temporal_sequence_basic', success, pattern_score)
        
        return {'pattern_recognized': success, 'score': pattern_score}
    
    def _recognize_pattern(self) -> float:
        """Recognize patterns in current sequence"""
        if len(self.current_sequence) < 3:
            return 0.0
        
        # Simple repetition detection
        events = [e['event'] for e in self.current_sequence]
        pattern_score = 0.0
        
        # Check for immediate repetitions
        for i in range(len(events) - 1):
            if events[i] == events[i + 1]:
                pattern_score += 0.2
        
        # Check for alternating patterns
        if len(events) >= 4:
            if events[0] == events[2] and events[1] == events[3]:
                pattern_score += 0.4
        
        return min(pattern_score, 1.0)

class SpatialRelationships:
    """Understanding basic spatial concepts: near/far, up/down, inside/outside"""
    
    def __init__(self):
        self.spatial_concepts = {
            'near': [], 'far': [], 'up': [], 'down': [],
            'inside': [], 'outside': [], 'left': [], 'right': []
        }
    
    def learn_spatial_relationship(self, object1: str, object2: str, relationship: str):
        """Learn spatial relationship between objects"""
        if relationship in self.spatial_concepts:
            self.spatial_concepts[relationship].append((object1, object2, time.time()))
            
            confidence = min(len(self.spatial_concepts[relationship]) * 0.1, 1.0)
            success = confidence > 0.3
            
            baby_tracker.record_learning_attempt('spatial_relationships', success, confidence)
            
            return {'learned': success, 'confidence': confidence}
        
        return {'learned': False, 'confidence': 0.0}

class AttentionFocusing:
    """Learning to focus attention on relevant stimuli"""
    
    def __init__(self):
        self.attention_targets = []
        self.focus_duration = []
        
    def focus_on_stimulus(self, stimulus: str, duration: float):
        """Record attention focusing event"""
        self.attention_targets.append(stimulus)
        self.focus_duration.append(duration)
        
        # Keep recent history only
        if len(self.focus_duration) > 20:
            self.focus_duration = self.focus_duration[-20:]
            self.attention_targets = self.attention_targets[-20:]
        
        avg_duration = np.mean(self.focus_duration)
        success = avg_duration > 2.0  # 2 seconds minimum focus
        
        baby_tracker.record_learning_attempt('attention_focusing', success, avg_duration / 10.0)
        
        return {'average_focus_duration': avg_duration, 'good_attention': success}

class CuriosityDrivenExploration:
    """Drives exploration of novel stimuli"""
    
    def __init__(self):
        self.explored_items = set()
        self.exploration_count = defaultdict(int)
        
    def explore_item(self, item: str, novelty_score: float):
        """Record exploration of an item"""
        self.exploration_count[item] += 1
        self.explored_items.add(item)
        
        # Curiosity satisfied when item explored multiple times
        exploration_satisfaction = min(self.exploration_count[item] * 0.25, 1.0)
        
        # Novelty drives initial exploration
        curiosity_score = novelty_score * (1.0 - exploration_satisfaction * 0.5)
        
        success = curiosity_score > 0.3
        baby_tracker.record_learning_attempt('curiosity_exploration', success, curiosity_score)
        
        return {
            'items_explored': len(self.explored_items),
            'curiosity_score': curiosity_score,
            'exploration_count': self.exploration_count[item]
        }

class ImitationLearningBasic:
    """Basic mimicking of observed actions"""
    
    def __init__(self):
        self.observed_actions = []
        self.imitated_actions = []
        
    def observe_action(self, action: str, complexity: float):
        """Observe an action for potential imitation"""
        self.observed_actions.append({
            'action': action,
            'complexity': complexity,
            'timestamp': time.time()
        })
        
        return {'observed': True, 'action': action}
    
    def attempt_imitation(self, action: str, success_rate: float):
        """Attempt to imitate an observed action"""
        # Check if action was observed
        observed = any(a['action'] == action for a in self.observed_actions)
        
        if observed:
            self.imitated_actions.append({
                'action': action,
                'success_rate': success_rate,
                'timestamp': time.time()
            })
            
            success = success_rate > 0.5
            baby_tracker.record_learning_attempt('imitation_learning_basic', success, success_rate)
            
            return {'imitated': success, 'success_rate': success_rate}
        
        return {'imitated': False, 'reason': 'Action not observed'}

class RewardAssociation:
    """Associates actions with positive/negative outcomes"""
    
    def __init__(self):
        self.action_rewards = defaultdict(list)
        
    def record_action_outcome(self, action: str, reward: float):
        """Record outcome of an action"""
        self.action_rewards[action].append(reward)
        
        # Keep recent history
        if len(self.action_rewards[action]) > 10:
            self.action_rewards[action] = self.action_rewards[action][-10:]
        
        avg_reward = np.mean(self.action_rewards[action])
        learning_strength = abs(avg_reward)  # Strong positive or negative learning
        
        success = learning_strength > 0.3
        baby_tracker.record_learning_attempt('reward_association', success, learning_strength)
        
        return {
            'action': action,
            'average_reward': avg_reward,
            'learning_strength': learning_strength,
            'learned_association': success
        }

# === BABY BRAIN ALGORITHM CONTROLLER ===
class BabyBrainController:
    """
    Central controller for all baby brain algorithms
    Manages algorithm selection and tracks overall development
    """
    
    def __init__(self):
        self.algorithms = {
            'cross_modal_learning': CrossModalBabyLearning(),
            'nursery_pattern_memory': NurseryPatternMemory(),
            'blur_tolerance': BlurToleranceProcessing(),
            'voice_familiarity': VoiceFamiliarityLearning(),
            'color_shape_association': ColorShapeAssociation(),
            'movement_tracking': MovementTracking(),
            'face_detection_simple': FaceDetectionSimple(),
            'emotional_tone_detection': EmotionalToneDetection(),
            'object_permanence': ObjectPermanence(),
            'cause_effect_simple': CauseEffectSimple(),
            'temporal_sequence_basic': TemporalSequenceBasic(),
            'spatial_relationships': SpatialRelationships(),
            'attention_focusing': AttentionFocusing(),
            'curiosity_exploration': CuriosityDrivenExploration(),
            'imitation_learning_basic': ImitationLearningBasic(),
            'reward_association': RewardAssociation()
        }
        
        self.development_phase = 'newborn'  # newborn, infant, early_toddler
        self.active_algorithms = set()
        
    def activate_algorithms_for_phase(self, phase: str) -> Dict[str, Any]:
        """Activate appropriate algorithms for development phase"""
        valid_phases = ['newborn', 'infant', 'early_toddler']
        
        if phase not in valid_phases:
            return {
                'success': False, 
                'error': f'Invalid phase: {phase}. Valid phases: {valid_phases}'
            }
        
        old_phase = self.development_phase
        self.development_phase = phase
        
        if phase == 'newborn':
            # 0-3 months: basic sensory processing
            self.active_algorithms = {
                'blur_tolerance', 'voice_familiarity', 'face_detection_simple',
                'emotional_tone_detection', 'cross_modal_learning'
            }
        elif phase == 'infant':
            # 3-12 months: more complex processing
            self.active_algorithms = {
                'blur_tolerance', 'voice_familiarity', 'face_detection_simple',
                'emotional_tone_detection', 'cross_modal_learning',
                'nursery_pattern_memory', 'movement_tracking', 'object_permanence',
                'cause_effect_simple', 'attention_focusing'
            }
        elif phase == 'early_toddler':
            # 12+ months: all algorithms active
            self.active_algorithms = set(self.algorithms.keys())
        
        # Log the transition
        if old_phase != phase:
            print(f"🧠 Development phase transition: {old_phase} → {phase}")
            print(f"⚡ Activated {len(self.active_algorithms)} algorithms")
            
            # Record this milestone in the tracker
            baby_tracker.development_timeline.append({
                'event': 'phase_transition',
                'old_phase': old_phase,
                'new_phase': phase,
                'algorithms_activated': len(self.active_algorithms),
                'timestamp': time.time()
            })
        
        return {
            'success': True,
            'phase': phase,
            'algorithms_activated': len(self.active_algorithms),
            'active_algorithms': list(self.active_algorithms)
        }

# === BABY BRAIN ALGORITHMS - COMPLETION ===
# This completes the BabyBrainController and adds integration methods
# Append this to the existing algorithm_baby_brain.py file

    def get_algorithm(self, name: str):
        """Get specific algorithm instance"""
        return self.algorithms.get(name)
    
    def process_multimodal_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through active baby brain algorithms"""
        results = {}
        
        # Route inputs to appropriate algorithms based on modality
        for modality, data in input_data.items():
            if modality == 'visual' and data is not None:
                results.update(self._process_visual_input(data))
            elif modality == 'audio' and data is not None:
                results.update(self._process_audio_input(data))
            elif modality == 'tactile' and data is not None:
                results.update(self._process_tactile_input(data))
        
        # Cross-modal processing
        if len(input_data) > 1:
            results['cross_modal'] = self._process_cross_modal(input_data)
        
        # Update development progress
        self._check_development_progress()
        
        return results
    
    def _process_visual_input(self, visual_data: np.ndarray) -> Dict[str, Any]:
        """Process visual input through relevant algorithms"""
        results = {}
        
        if 'blur_tolerance' in self.active_algorithms:
            results['blur_tolerance'] = self.algorithms['blur_tolerance'].process_blurry_image(visual_data)
        
        if 'face_detection_simple' in self.active_algorithms:
            results['face_detection'] = self.algorithms['face_detection_simple'].detect_face(visual_data)
        
        if 'movement_tracking' in self.active_algorithms:
            results['movement_tracking'] = self.algorithms['movement_tracking'].track_movement(visual_data)
        
        if 'nursery_pattern_memory' in self.active_algorithms:
            # Try to recognize visual patterns
            for category in ['shape', 'color']:
                recognition = self.algorithms['nursery_pattern_memory'].recognize_pattern(category, visual_data)
                results[f'pattern_{category}'] = recognition
        
        return results
    
    def _process_audio_input(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process audio input through relevant algorithms"""
        results = {}
        
        if 'voice_familiarity' in self.active_algorithms:
            results['voice_recognition'] = self.algorithms['voice_familiarity'].recognize_voice(audio_data)
        
        if 'emotional_tone_detection' in self.active_algorithms:
            results['emotion_detection'] = self.algorithms['emotional_tone_detection'].detect_emotion(audio_data)
        
        if 'nursery_pattern_memory' in self.active_algorithms:
            # Try to recognize sound patterns
            recognition = self.algorithms['nursery_pattern_memory'].recognize_pattern('sound', audio_data)
            results['pattern_sound'] = recognition
        
        return results
    
    def _process_tactile_input(self, tactile_data: Dict[str, float]) -> Dict[str, Any]:
        """Process tactile input through relevant algorithms"""
        results = {}
        
        if 'attention_focusing' in self.active_algorithms:
            # Tactile input can drive attention
            focus_result = self.algorithms['attention_focusing'].focus_on_stimulus(
                'tactile', tactile_data.get('intensity', 0.5) * 2.0
            )
            results['tactile_attention'] = focus_result
        
        return results
    
    def _process_cross_modal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cross-modal associations"""
        if 'cross_modal_learning' not in self.active_algorithms:
            return {}
        
        cross_modal = self.algorithms['cross_modal_learning']
        
        # Add each modality input to cross-modal learning
        associations = []
        current_time = time.time()
        
        for modality, data in input_data.items():
            if data is not None:
                event, connections = cross_modal.add_sensory_input(modality, data, current_time)
                associations.extend(connections)
        
        return {'associations': associations, 'total_connections': len(associations)}
    
    def _check_development_progress(self):
        """Check if baby should progress to next development phase"""
        report = baby_tracker.get_development_report()
        overall_success = report['overall_success_rate']
        
        # Development progression criteria
        if self.development_phase == 'newborn' and overall_success > 0.6:
            self.activate_algorithms_for_phase('infant')
            print(f"🎉 Development milestone: Progressed to infant phase! Success rate: {overall_success:.2f}")
        elif self.development_phase == 'infant' and overall_success > 0.7:
            self.activate_algorithms_for_phase('early_toddler')
            print(f"🎉 Development milestone: Progressed to early toddler phase! Success rate: {overall_success:.2f}")
    
    def learn_from_labeled_example(self, input_data: Dict[str, Any], labels: Dict[str, Any]):
        """Learn from a labeled training example"""
        results = {}
        
        # Visual learning
        if 'visual' in input_data and 'visual_labels' in labels:
            visual_labels = labels['visual_labels']
            
            if 'face' in visual_labels and 'face_detection_simple' in self.active_algorithms:
                # This is training data for face detection - record success
                detection_result = self.algorithms['face_detection_simple'].detect_face(input_data['visual'])
                success = detection_result['face_detected'] == visual_labels['face']
                baby_tracker.record_learning_attempt('face_detection_simple', success, detection_result['confidence'])
            
            if 'color' in visual_labels and 'color_shape_association' in self.active_algorithms:
                color = visual_labels['color']
                shape = visual_labels.get('shape', 'unknown')
                self.algorithms['color_shape_association'].learn_association(color, shape)
            
            if 'pattern_category' in visual_labels and 'nursery_pattern_memory' in self.active_algorithms:
                category = visual_labels['pattern_category']
                label = visual_labels.get('pattern_label', None)
                self.algorithms['nursery_pattern_memory'].learn_pattern(category, input_data['visual'], label)
        
        # Audio learning
        if 'audio' in input_data and 'audio_labels' in labels:
            audio_labels = labels['audio_labels']
            
            if 'voice_id' in audio_labels and 'voice_familiarity' in self.active_algorithms:
                voice_id = audio_labels['voice_id']
                voice_label = audio_labels.get('voice_label', voice_id)
                self.algorithms['voice_familiarity'].learn_voice(voice_id, input_data['audio'], voice_label)
            
            if 'emotion' in audio_labels and 'emotional_tone_detection' in self.active_algorithms:
                # This is training data for emotion detection
                emotion_result = self.algorithms['emotional_tone_detection'].detect_emotion(input_data['audio'])
                success = emotion_result['emotion'] == audio_labels['emotion']
                baby_tracker.record_learning_attempt('emotional_tone_detection', success, emotion_result['confidence'])
        
        # Cross-modal learning
        if len(input_data) > 1 and 'cross_modal_learning' in self.active_algorithms:
            # Let cross-modal learning find temporal associations
            self._process_cross_modal(input_data)
        
        return results
    
    def get_development_report(self) -> Dict[str, Any]:
        """Get comprehensive development report"""
        base_report = baby_tracker.get_development_report()
        
        # Add controller-specific information
        controller_report = {
            'development_phase': self.development_phase,
            'active_algorithms': list(self.active_algorithms),
            'total_algorithms': len(self.algorithms),
            'algorithm_performance': {},
            'development_milestones': self._calculate_milestones()
        }
        
        # Get performance for each active algorithm
        for algo_name in self.active_algorithms:
            if algo_name in baby_tracker.metrics:
                metric = baby_tracker.metrics[algo_name]
                controller_report['algorithm_performance'][algo_name] = {
                    'success_rate': metric.success_rate,
                    'learning_status': metric.learning_status.value,
                    'attempts': metric.total_attempts
                }
        
        return {**base_report, **controller_report}
    
    def _calculate_milestones(self) -> Dict[str, bool]:
        """Calculate development milestones achieved"""
        milestones = {}
        
        # Basic sensory milestones
        milestones['recognizes_faces'] = self._check_algorithm_success('face_detection_simple', 0.6)
        milestones['recognizes_voices'] = self._check_algorithm_success('voice_familiarity', 0.6)
        milestones['processes_blur'] = self._check_algorithm_success('blur_tolerance', 0.5)
        
        # Cognitive milestones
        milestones['understands_object_permanence'] = self._check_algorithm_success('object_permanence', 0.5)
        milestones['learns_cause_effect'] = self._check_algorithm_success('cause_effect_simple', 0.5)
        milestones['forms_associations'] = self._check_algorithm_success('color_shape_association', 0.4)
        
        # Social milestones
        milestones['detects_emotions'] = self._check_algorithm_success('emotional_tone_detection', 0.5)
        milestones['imitates_actions'] = self._check_algorithm_success('imitation_learning_basic', 0.4)
        
        # Advanced milestones
        milestones['cross_modal_learning'] = self._check_algorithm_success('cross_modal_learning', 0.4)
        milestones['focused_attention'] = self._check_algorithm_success('attention_focusing', 0.5)
        
        return milestones
    
    def _check_algorithm_success(self, algo_name: str, threshold: float) -> bool:
        """Check if algorithm has achieved success threshold"""
        if algo_name in baby_tracker.metrics:
            return baby_tracker.metrics[algo_name].success_rate >= threshold
        return False
    
    def save_development_state(self, filepath: str):
        """Save current development state to file"""
        state = {
            'development_phase': self.development_phase,
            'active_algorithms': list(self.active_algorithms),
            'metrics': {},
            'algorithm_states': {}
        }
        
        # Save metrics
        for name, metric in baby_tracker.metrics.items():
            state['metrics'][name] = {
                'total_attempts': metric.total_attempts,
                'successful_attempts': metric.successful_attempts,
                'confidence_scores': metric.confidence_scores,
                'connection_strength': metric.connection_strength,
                'pattern_diversity': metric.pattern_diversity
            }
        
        # Save algorithm-specific states
        for name, algorithm in self.algorithms.items():
            if hasattr(algorithm, '__dict__'):
                # Save algorithm state (simplified)
                try:
                    state['algorithm_states'][name] = str(algorithm.__dict__)
                except:
                    state['algorithm_states'][name] = 'state_not_serializable'
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_development_state(self, filepath: str):
        """Load development state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.development_phase = state.get('development_phase', 'newborn')
            self.active_algorithms = set(state.get('active_algorithms', []))
            
            # Restore metrics
            for name, metric_data in state.get('metrics', {}).items():
                baby_tracker.metrics[name] = BabyLearningMetric(
                    algorithm_name=name,
                    total_attempts=metric_data.get('total_attempts', 0),
                    successful_attempts=metric_data.get('successful_attempts', 0),
                    confidence_scores=metric_data.get('confidence_scores', []),
                    connection_strength=metric_data.get('connection_strength', 0.0),
                    pattern_diversity=metric_data.get('pattern_diversity', 0.0)
                )
            
            print(f"✅ Development state loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load development state: {e}")
            return False
    
    def reset_development(self):
        """Reset all learning progress - start fresh"""
        baby_tracker.metrics.clear()
        baby_tracker.cross_modal_connections.clear()
        baby_tracker.pattern_memory.clear()
        baby_tracker.development_timeline.clear()
        
        # Reset algorithm states
        for algorithm in self.algorithms.values():
            if hasattr(algorithm, '__init__'):
                algorithm.__init__()
        
        self.development_phase = 'newborn'
        self.activate_algorithms_for_phase('newborn')
        
        print("🔄 Baby brain development reset to newborn phase")
    
    def run_development_test(self) -> Dict[str, Any]:
        """Run a comprehensive test of baby brain development"""
        print("🧠 Running Baby Brain Development Test...")
        
        # Test data
        test_visual = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        test_audio = np.random.randn(1000).astype(np.float32)
        test_tactile = {'pressure': 0.5, 'temperature': 0.7}
        
        # Test single modality processing
        visual_results = self._process_visual_input(test_visual)
        audio_results = self._process_audio_input(test_audio)
        tactile_results = self._process_tactile_input(test_tactile)
        
        # Test multimodal processing
        multimodal_input = {
            'visual': test_visual,
            'audio': test_audio,
            'tactile': test_tactile
        }
        multimodal_results = self.process_multimodal_input(multimodal_input)
        
        # Test learning from labeled examples
        labels = {
            'visual_labels': {'face': True, 'color': (255, 0, 0), 'shape': 'circle'},
            'audio_labels': {'voice_id': 'mama', 'emotion': 'happy'}
        }
        learning_results = self.learn_from_labeled_example(multimodal_input, labels)
        
        # Generate report
        development_report = self.get_development_report()
        
        test_results = {
            'visual_processing': len(visual_results),
            'audio_processing': len(audio_results),
            'tactile_processing': len(tactile_results),
            'multimodal_processing': len(multimodal_results),
            'learning_capability': len(learning_results),
            'development_report': development_report,
            'test_passed': development_report['overall_success_rate'] > 0.0
        }
        
        print(f"✅ Test completed! Overall success rate: {development_report['overall_success_rate']:.2f}")
        print(f"🎯 Active algorithms: {len(self.active_algorithms)}")
        print(f"📊 Development phase: {self.development_phase}")
        
        return test_results

# === GLOBAL BABY BRAIN INSTANCE ===
# Create global instance for easy access
baby_brain = BabyBrainController()

# === UTILITY FUNCTIONS ===
def get_baby_brain_status() -> Dict[str, Any]:
    """Get current status of baby brain system"""
    return baby_brain.get_development_report()

def train_baby_brain_example(visual_data=None, audio_data=None, tactile_data=None, labels=None):
    """Train baby brain with a single example"""
    input_data = {}
    if visual_data is not None:
        input_data['visual'] = visual_data
    if audio_data is not None:
        input_data['audio'] = audio_data
    if tactile_data is not None:
        input_data['tactile'] = tactile_data
    
    if labels:
        return baby_brain.learn_from_labeled_example(input_data, labels)
    else:
        return baby_brain.process_multimodal_input(input_data)

def reset_baby_brain():
    """Reset baby brain to initial state"""
    baby_brain.reset_development()

def run_baby_brain_test():
    """Run comprehensive baby brain test"""
    return baby_brain.run_development_test()

# === ALGORITHM NAMES REFERENCE ===
BABY_BRAIN_ALGORITHMS = [
    'cross_modal_learning',
    'nursery_pattern_memory', 
    'blur_tolerance',
    'voice_familiarity',
    'color_shape_association',
    'movement_tracking',
    'face_detection_simple',
    'emotional_tone_detection',
    'object_permanence',
    'cause_effect_simple',
    'temporal_sequence_basic',
    'spatial_relationships',
    'attention_focusing',
    'curiosity_exploration',
    'imitation_learning_basic',
    'reward_association'
]

if __name__ == "__main__":
    # Example usage and testing
    print("🧠 Baby Brain Algorithms System Initialized")
    print(f"📚 Total algorithms available: {len(BABY_BRAIN_ALGORITHMS)}")
    print(f"🎯 Current development phase: {baby_brain.development_phase}")
    print(f"⚡ Active algorithms: {len(baby_brain.active_algorithms)}")
    
    # Run development test
    test_results = run_baby_brain_test()
    print(f"\n🎉 Baby Brain Test Results: {test_results['test_passed']}")