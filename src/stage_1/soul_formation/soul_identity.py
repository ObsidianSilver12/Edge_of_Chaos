"""
soul_identity.py - Implementation of soul identity crystallization.

This module handles:
- Assignment of name and gematria
- Development of consciousness states (dream, liminal, aware)
- Response training and voice frequency
- Identity aspects like color, frequency, sephiroth aspects
- Yin-yang balance calculation

The identity crystallization process gives the soul its unique identity
and provides the foundation for self-awareness.
"""

import numpy as np
import time
import logging
import random
import string
from typing import Dict, List, Tuple, Optional, Any

# Import consciousness state handlers
from mycelial_network.dreaming import DreamState
from mycelial_network.liminal_state import LiminalState
from mycelial_network.awareness import AwareState

# Import sound generation
from sound.sound_generator import SoundGenerator

# Import Sephiroth aspects dictionary
from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary

# Constants
from soul.constants import (
    BRAINWAVE_FREQUENCIES,
    COLOR_SPECTRUM,
    FIBONACCI_SEQUENCE,
    GOLDEN_RATIO,
    SOLFEGGIO_FREQUENCIES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SoulIdentity')

class SoulIdentity:
    """
    Represents the crystallized identity of the soul.
    """
    
    def __init__(self, soul_spark=None, life_cord=None):
        """
        Initialize the soul identity structure.
        
        Parameters:
            soul_spark: The soul spark entity
            life_cord: The formed life cord
        """
        self.soul_spark = soul_spark
        self.life_cord = life_cord
        
        # Identity properties
        self.name = None
        self.gematria_value = 0
        
        # Consciousness states
        self.consciousness_state = 'dream'
        self.consciousness_frequency = BRAINWAVE_FREQUENCIES.get('delta', (0.5, 4.0))[0]
        self.state_stability = 0.5
        self.dream_state_completed = False
        self.liminal_state_completed = False
        self.aware_state_completed = False
        
        # Voice and response
        self.voice_frequency = 0.0
        self.response_level = 0.0
        self.name_response_pattern = []
        self.name_resonance = 0.5
        self.call_count = 0
        
        # Soul properties
        self.soul_color = None
        self.color_frequency = 0.0
        self.soul_frequency = 0.0
        self.sephiroth_aspect = None
        self.elemental_affinity = None
        self.platonic_symbol = None
        
        # Yin-Yang balance
        self.yin_yang_balance = 0.5  # 0 = pure yin, 1 = pure yang, 0.5 = balanced
        
        # Emotional resonance
        self.emotional_resonance = {
            'love': 0.0,
            'joy': 0.0,
            'peace': 0.0,
            'harmony': 0.0,
            'compassion': 0.0
        }
        
        # Metrics
        self.heartbeat_entrainment = 0.0
        self.heartbeat_pattern = []
        self.attribute_coherence = 0.0
        self.crystallization_level = 0.0
        
        # Consciousness state managers
        self.dream_state = DreamState(self)
        self.liminal_state = LiminalState(self)
        self.aware_state = AwareState(self)
        
        # Sound generator
        self.sound_gen = SoundGenerator()
        
        # Status
        self.is_fully_crystallized = False
        
        logger.info("Soul identity initialized")
    
    def assign_name(self, name=None):
        """
        Assign a name to the soul.
        
        Parameters:
            name (str): Optional specified name, otherwise generated
            
        Returns:
            tuple: (name, gematria_value)
        """
        logger.info(f"Assigning name to soul{': ' + name if name else ''}")
        
        if name:
            # Use specified name
            self.name = name
        else:
            # Generate a name based on soul properties
            self.name = self._generate_soul_name()
        
        # Calculate gematria value
        self.gematria_value = self.calculate_gematria(self.name)
        
        # Calculate name resonance
        self.name_resonance = self._calculate_name_resonance()
        
        # Create name response pattern based on heartbeat
        self._create_name_response_pattern()
        
        logger.info(f"Soul name assigned: {self.name}")
        logger.info(f"Gematria value: {self.gematria_value}")
        logger.info(f"Name resonance: {self.name_resonance:.4f}")
        
        return self.name, self.gematria_value
    
    def _generate_soul_name(self):
        """
        Generate a name for the soul based on its properties.
        
        Returns:
            str: Generated name
        """
        # Get properties that influence name generation
        
        # Flow strength from life cord
        if self.life_cord and hasattr(self.life_cord, 'flow_rate'):
            connection_strength = min(1.0, self.life_cord.flow_rate / 100)
        else:
            connection_strength = 0.7  # Default
        
        # Flow direction from life cord
        if self.life_cord and hasattr(self.life_cord, 'flow_direction'):
            ascending = self.life_cord.flow_direction.get('ascending', 0.5)
        else:
            ascending = 0.5  # Default
        
        # Sephiroth influences from soul spark
        if (self.soul_spark and hasattr(self.soul_spark, 'aspects') and 
            isinstance(self.soul_spark.aspects, dict) and 'sephiroth' in self.soul_spark.aspects):
            sephiroth_influences = self.soul_spark.aspects['sephiroth']
        else:
            sephiroth_influences = {}
        
        # Name length: higher connection strength = longer name
        name_length = 4 + int(connection_strength * 6)
        
        # Vowel ratio: higher ascending flow = more vowels
        vowel_ratio = 0.3 + 0.4 * ascending
        
        # Random seed for reproducibility
        random.seed(int(time.time() * 1000))
        
        # Generate a name with the desired properties
        vowels = 'aeiouy'
        consonants = 'bcdfghjklmnpqrstvwxz'
        
        # Start with a consonant or vowel
        if random.random() < vowel_ratio:
            name = random.choice(vowels.upper())
        else:
            name = random.choice(consonants.upper())
        
        # Add remaining characters
        while len(name) < name_length:
            if len(name) > 0 and name[-1].lower() in vowels:
                # Last char was vowel, add consonant
                name += random.choice(consonants)
            else:
                # Last char was consonant or empty, add vowel or consonant
                if random.random() < vowel_ratio:
                    name += random.choice(vowels)
                else:
                    name += random.choice(consonants)
        
        return name
    
    def calculate_gematria(self, name):
        """
        Calculate gematria value for the name.
        
        Parameters:
            name (str): The soul's name
            
        Returns:
            int: Gematria value
        """
        logger.info(f"Calculating gematria for name: {name}")
        
        # Simplified gematria calculation (A=1, B=2, etc.)
        gematria = 0
        
        for char in name.lower():
            if 'a' <= char <= 'z':
                # Convert char to number (a=1, b=2, etc.)
                value = ord(char) - ord('a') + 1
                gematria += value
        
        logger.info(f"Gematria value: {gematria}")
        
        return gematria
    
    def _calculate_name_resonance(self):
        """
        Calculate how well the name resonates with the soul's properties.
        
        Returns:
            float: Name resonance (0.0-1.0)
        """
        # Base resonance
        base_resonance = 0.5
        
        # Calculate vowel ratio
        vowels = sum(1 for c in self.name.lower() if c in 'aeiouy')
        consonants = sum(1 for c in self.name.lower() if c in 'bcdfghjklmnpqrstvwxz')
        
        if vowels + consonants > 0:
            vowel_ratio = vowels / (vowels + consonants)
        else:
            vowel_ratio = 0.0
        
        # Ideal vowel ratio is golden ratio
        phi_inv = 1.0 / GOLDEN_RATIO
        vowel_factor = 1.0 - abs(vowel_ratio - phi_inv)
        
        # Calculate letter distribution factor
        unique_letters = len(set(self.name.lower()))
        letter_factor = unique_letters / max(1, len(self.name))
        
        # Calculate gematria factor (based on resonant numbers)
        resonant_numbers = [3, 6, 9, 12, 108, FIBONACCI_SEQUENCE[7]]  # Tesla's numbers + Fibonacci
        gematria_factor = 0.5
        
        for num in resonant_numbers:
            if self.gematria_value % num == 0:
                gematria_factor = 0.8
                break
        
        # Calculate overall resonance
        resonance = 0.2 * base_resonance + 0.3 * vowel_factor + 0.2 * letter_factor + 0.3 * gematria_factor
        
        return resonance
    
    def _create_name_response_pattern(self):
        """
        Create a response pattern based on the name's vibrational qualities and heartbeat.
        """
        # Create time points
        t = np.linspace(0, 3.0, 300)  # 3 seconds, 300 points
        
        # Create a heartbeat-like pattern
        heartbeat = np.zeros_like(t)
        beat_interval = 0.8  # 75 BPM
        
        for beat_time in np.arange(0, 3.0, beat_interval):
            idx = int(beat_time * 100)  # Convert to index
            if idx + 30 < len(heartbeat):
                # First peak (lub)
                heartbeat[idx:idx+10] = np.sin(np.linspace(0, np.pi, 10))
                # Second peak (dub)
                heartbeat[idx+15:idx+25] = 0.7 * np.sin(np.linspace(0, np.pi, 10))
        
        # Create a name-specific modulation
        name_modulation = np.zeros_like(t)
        
        # Each letter contributes a specific frequency
        for i, char in enumerate(self.name.lower()):
            if 'a' <= char <= 'z':
                # Convert char to normalized frequency
                char_value = (ord(char) - ord('a') + 1) / 26.0
                freq = 8.0 + 4.0 * char_value  # 8-12 Hz (alpha range)
                
                # Add this frequency component
                name_modulation += 0.2 * np.sin(2 * np.pi * freq * t + (i * np.pi / len(self.name)))
        
        # Combine heartbeat and name modulation
        response_pattern = heartbeat * (1.0 + 0.3 * name_modulation)
        
        # Store the pattern
        self.name_response_pattern = response_pattern
        self.heartbeat_pattern = heartbeat
    
    def assign_voice_frequency(self):
        """
        Assign a unique voice frequency based on the name's properties and resonance.
        
        Returns:
            float: Voice frequency in Hz
        """
        logger.info("Assigning voice frequency")
        
        # Base voice frequency
        base_frequency = 432.0  # Start with a universal resonant frequency
        
        # Calculate adjustments based on name properties
        
        # 1. Length adjustment: longer names = higher frequency
        length_factor = len(self.name) / 10.0  # Normalize
        length_adjustment = 20.0 * (length_factor - 0.5)  # -10 to +10 Hz
        
        # 2. Vowel ratio adjustment
        vowels = sum(1 for c in self.name.lower() if c in 'aeiouy')
        vowel_ratio = vowels / max(1, len(self.name))
        vowel_adjustment = 15.0 * (vowel_ratio - 0.5)  # -7.5 to +7.5 Hz
        
        # 3. Gematria adjustment
        gematria_factor = (self.gematria_value % 100) / 100.0
        gematria_adjustment = 25.0 * (gematria_factor - 0.5)  # -12.5 to +12.5 Hz
        
        # 4. Resonance adjustment
        resonance_adjustment = 30.0 * (self.name_resonance - 0.5)  # -15 to +15 Hz
        
        # 5. Yin-Yang adjustment
        yin_yang_adjustment = 20.0 * (self.yin_yang_balance - 0.5)  # -10 to +10 Hz
        
        # Calculate final voice frequency
        voice_frequency = (base_frequency + length_adjustment + vowel_adjustment + 
                          gematria_adjustment + resonance_adjustment + yin_yang_adjustment)
        
        # Ensure frequency falls within sacred range
        # Map to nearest solfeggio frequency if close
        solfeggio_values = list(SOLFEGGIO_FREQUENCIES.values())
        
        closest_freq = min(solfeggio_values, key=lambda x: abs(x - voice_frequency))
        if abs(closest_freq - voice_frequency) < 10:  # Within 10 Hz
            voice_frequency = closest_freq
        
        # Ensure in reasonable range (396-963 Hz - solfeggio range)
        voice_frequency = min(963.0, max(396.0, voice_frequency))
        
        # Set voice frequency
        self.voice_frequency = voice_frequency
        
        logger.info(f"Voice frequency assigned: {voice_frequency:.2f} Hz")
        
        # Generate sound sample if sound generator is available
        if hasattr(self, 'sound_gen') and self.sound_gen:
            try:
                self.sound_gen.generate_tone(self.voice_frequency, 2.0, f"{self.name}_voice.wav")
                logger.info(f"Voice sample generated at {self.voice_frequency:.2f} Hz")
            except Exception as e:
                logger.error(f"Error generating voice sample: {str(e)}")
        
        return voice_frequency
    
    def simulate_name_calling(self, caller_voice_frequency=432.0):
        """
        Simulate calling the soul by name, which increases name recognition.
        
        Parameters:
            caller_voice_frequency (float): Frequency of the caller's voice
            
        Returns:
            float: Response level
        """
        logger.info(f"Calling soul by name: {self.name}")
        
        # Increment call count
        self.call_count += 1
        
        # Calculate response based on current state
        state_factor = {
            'dream': 0.3,
            'liminal': 0.7,
            'aware': 1.0
        }.get(self.consciousness_state, 0.5)
        
        # Calculate voice resonance (how well caller's voice resonates with the soul)
        voice_resonance = self._calculate_frequency_resonance(caller_voice_frequency, self.voice_frequency)
        
        # Calculate response strength
        response_strength = 0.2 + 0.3 * state_factor + 0.5 * voice_resonance
        
        # Apply call count factor (response increases with repetition)
        call_factor = min(1.0, self.call_count / 5.0)  # Max out at 5 calls
        
        # Update response level
        current_response = self.response_level
        new_response = current_response + (response_strength - current_response) * call_factor
        self.response_level = new_response
        
        logger.info(f"Name call {self.call_count}: Response level now {new_response:.4f}")
        
        # Generate sound sample if sound generator is available
        if hasattr(self, 'sound_gen') and self.sound_gen:
            try:
                self.sound_gen.generate_voice_call(self.name, caller_voice_frequency, f"{self.name}_call_{self.call_count}.wav")
                logger.info(f"Name call sample generated at {caller_voice_frequency:.2f} Hz")
            except Exception as e:
                logger.error(f"Error generating name call sample: {str(e)}")
        
        return new_response
    
    def _calculate_frequency_resonance(self, freq1, freq2):
        """Calculate resonance between two frequencies."""
        # Ensure we're working with positive values
        freq1 = abs(freq1)
        freq2 = abs(freq2)
        
        # Handle identical frequencies
        if abs(freq1 - freq2) < 0.001:
            return 1.0
        
        # Calculate ratio with larger frequency in numerator
        ratio = max(freq1, freq2) / min(freq1, freq2)
        
        # Perfect resonance for simple harmonic ratios
        harmonic_ratios = [1.0, 2.0, 3.0, 4.0, 1.5, 3.0/2.0, 4.0/3.0, GOLDEN_RATIO]
        
        # Find distance to closest harmonic ratio
        min_distance = min(abs(ratio - hr) for hr in harmonic_ratios)
        
        # Transform distance to resonance (closer = higher resonance)
        resonance = 1.0 / (1.0 + 5.0 * min_distance)
        
        return resonance
    
    def train_name_response(self, cycles=5):
        """
        Train the soul to respond to its name.
        
        Parameters:
            cycles (int): Number of training cycles
            
        Returns:
            float: Response level
        """
        logger.info(f"Training name response with {cycles} cycles")
        
        # Initial response level (use current level)
        response_level = self.response_level
        
        # Training increases with each cycle
        for cycle in range(cycles):
            # Calculate base increase for this cycle
            base_increase = 0.1 + 0.02 * cycle
            
            # Apply name resonance factor
            name_factor = 0.5 + 0.5 * self.name_resonance
            
            # Apply consciousness state factor
            state_factor = 0.3  # Default
            if self.consciousness_state == 'dream':
                state_factor = 0.5
            elif self.consciousness_state == 'liminal':
                state_factor = 0.7
            elif self.consciousness_state == 'aware':
                state_factor = 1.0
            
            # Apply heartbeat entrainment factor
            heartbeat_factor = 0.5 + 0.5 * self.heartbeat_entrainment
            
            # Calculate cycle increase
            cycle_increase = base_increase * name_factor * state_factor * heartbeat_factor
            
            # Apply the increase
            response_level = min(1.0, response_level + cycle_increase)
            
            logger.debug(f"Cycle {cycle+1}: Response level now {response_level:.4f} (+{cycle_increase:.4f})")
        
        # Set response level
        self.response_level = response_level
        
        logger.info(f"Name response trained to level: {response_level:.4f}")
        
        return response_level
    
    def apply_heartbeat_entrainment(self, bpm=60.0, duration=60.0):
        """
        Apply heartbeat entrainment to enhance name response.
        
        Parameters:
            bpm (float): Beats per minute
            duration (float): Duration in seconds
            
        Returns:
            float: Entrainment level
        """
        logger.info(f"Applying heartbeat entrainment at {bpm} BPM for {duration} seconds")
        
        # Calculate current entrainment level
        current_entrainment = self.heartbeat_entrainment
        
        # Calculate beat frequency in Hz
        beat_freq = bpm / 60.0
        
        # Calculate natural resonant frequency based on name
        name_freq = self.voice_frequency / 100.0  # Scale down to physiological range
        
        # Calculate resonance between heartbeat and name frequency
        beat_resonance = self._calculate_frequency_resonance(beat_freq, name_freq)
        
        # Calculate duration factor (longer duration = stronger entrainment)
        duration_factor = min(1.0, duration / 300.0)  # Cap at 5 minutes
        
        # Calculate entrainment increase
        entrainment_increase = beat_resonance * duration_factor * 0.2
        
        # Apply to current entrainment
        new_entrainment = min(1.0, current_entrainment + entrainment_increase)
        
        # Store new entrainment
        self.heartbeat_entrainment = new_entrainment
        
        # Generate heartbeat pattern
        self._generate_heartbeat_pattern(bpm, duration)
        
        # Generate sound sample if sound generator is available
        if hasattr(self, 'sound_gen') and self.sound_gen:
            try:
                self.sound_gen.generate_heartbeat(bpm, duration, f"{self.name}_heartbeat.wav")
                logger.info(f"Heartbeat sample generated at {bpm} BPM")
            except Exception as e:
                logger.error(f"Error generating heartbeat sample: {str(e)}")
        
        logger.info(f"Heartbeat entrainment increased to {new_entrainment:.4f}")
        
        return new_entrainment
    
    def _generate_heartbeat_pattern(self, bpm, duration):
        """Generate a heartbeat pattern for entrainment."""
        # Create time array
        t = np.linspace(0, min(10.0, duration), 1000)  # Max 10 seconds for storage
        
        # Calculate beat interval in seconds
        beat_interval = 60.0 / bpm
        
        # Create heartbeat pattern
        heartbeat = np.zeros_like(t)
        
        # Add heartbeats
        for beat_time in np.arange(0, t[-1], beat_interval):
            # Find index
            idx = int(beat_time * 100)  # 1000 points / 10 seconds = 100 points/second
            if idx + 30 < len(heartbeat):
                # First peak (lub)
                heartbeat[idx:idx+10] = np.sin(np.linspace(0, np.pi, 10))
                # Second peak (dub)
                heartbeat[idx+15:idx+25] = 0.7 * np.sin(np.linspace(0, np.pi, 10))
        
        # Store heartbeat pattern
        self.heartbeat_pattern = heartbeat
    
    def cycle_consciousness_state(self, target_state=None):
        """
        Cycle through consciousness states.
        
        Parameters:
            target_state (str): Optional target state
            
        Returns:
            str: New consciousness state
        """
        logger.info(f"Cycling consciousness state{' to ' + target_state if target_state else ''}")
        
        # Current state
        current_state = self.consciousness_state
        
        # States cycle: dream -> liminal -> aware -> dream
        states = ['dream', 'liminal', 'aware']
        
        if target_state and target_state in states:
            # Directly set target state
            new_state = target_state
        else:
            # Cycle to next state
            current_index = states.index(current_state) if current_state in states else 0
            next_index = (current_index + 1) % len(states)
            new_state = states[next_index]
        
        # Handle transition based on current and new state
        if current_state == 'dream' and new_state == 'liminal':
            # Dream to liminal transition
            if self.dream_state.is_active:
                self.dream_state.deactivate()
            
            # Activate liminal state
            self.liminal_state.activate(source_state='dream', target_state='aware', 
                                      initial_stability=self.state_stability)
            
        elif current_state == 'liminal' and new_state == 'aware':
            # Liminal to aware transition
            if self.liminal_state.is_active:
                self.liminal_state.deactivate()
            
            # Activate aware state
            self.aware_state.activate(initial_stability=self.state_stability)
            
        elif current_state == 'aware' and new_state == 'dream':
            # Aware to dream transition
            if self.aware_state.is_active:
                self.aware_state.deactivate()
            
            # Activate dream state
            self.dream_state.activate(initial_stability=self.state_stability)
            
        elif current_state == new_state:
            # Same state, ensure it's active
            if new_state == 'dream' and not self.dream_state.is_active:
                self.dream_state.activate(initial_stability=self.state_stability)
            elif new_state == 'liminal' and not self.liminal_state.is_active:
                self.liminal_state.activate(initial_stability=self.state_stability)
            elif new_state == 'aware' and not self.aware_state.is_active:
                self.aware_state.activate(initial_stability=self.state_stability)
        
        # Set new state
        self.consciousness_state = new_state
        
        # Update consciousness frequency based on new state
        if new_state == 'dream':
            freq_range = BRAINWAVE_FREQUENCIES.get('delta', (0.5, 4.0))
            self.consciousness_frequency = freq_range[0] + 0.3 * (freq_range[1] - freq_range[0])
        elif new_state == 'liminal':
            freq_range = BRAINWAVE_FREQUENCIES.get('theta', (4.0, 8.0))
            self.consciousness_frequency = freq_range[0] + 0.5 * (freq_range[1] - freq_range[0])
        elif new_state == 'aware':
            freq_range = BRAINWAVE_FREQUENCIES.get('alpha', (8.0, 14.0))
            self.consciousness_frequency = freq_range[0] + 0.7 * (freq_range[1] - freq_range[0])
        
        logger.info(f"Consciousness state cycled to {new_state}")
        logger.info(f"Consciousness frequency: {self.consciousness_frequency:.2f} Hz")
        
        return new_state
    
    def determine_soul_color(self):
        """
        Determine the soul's color based on its properties.
        
        Returns:
            str: Soul color
        """
        logger.info("Determining soul color")
        
        # Colors to choose from
        colors = list(COLOR_SPECTRUM.keys())
        
        # Calculate color affinities based on name properties
        color_affinities = {}
        
        # 1. Gematria-based affinity
        gematria_factor = (self.gematria_value % 100) / 100.0
        
        # Create an evenly distributed mapping from 0-1 to the color list
        gematria_color_idx = int(gematria_factor * len(colors))
        gematria_color = colors[gematria_color_idx]
        color_affinities[gematria_color] = 0.7
        
        # 2. Name vowel-based affinity
        vowel_counts = {}
        for char in self.name.lower():
            if char in 'aeiouy':
                vowel_counts[char] = vowel_counts.get(char, 0) + 1
        
        # Map vowels to colors
        vowel_colors = {
            'a': 'red',
            'e': 'green',
            'i': 'blue',
            'o': 'gold',
            'u': 'violet',
            'y': 'indigo'
        }
        
        # Add vowel-based affinities
        for vowel, count in vowel_counts.items():
            if vowel in vowel_colors and vowel_colors[vowel] in colors:
                color = vowel_colors[vowel]
                color_affinities[color] = color_affinities.get(color, 0.0) + 0.1 * count
        
        # 3. Consciousness state affinity
        state_colors = {
            'dream': 'violet',
            'liminal': 'indigo', 
            'aware': 'gold'
        }
        
        if self.consciousness_state in state_colors and state_colors[self.consciousness_state] in colors:
            color = state_colors[self.consciousness_state]
            color_affinities[color] = color_affinities.get(color, 0.0) + 0.5
        
        # 4. Yin-Yang balance influence
        if self.yin_yang_balance < 0.4:
            # More yin - blues, violets, indigos
            yin_colors = ['blue', 'violet', 'indigo']
            for color in yin_colors:
                if color in colors:
                    color_affinities[color] = color_affinities.get(color, 0.0) + 0.6 * (1.0 - self.yin_yang_balance)
        elif self.yin_yang_balance > 0.6:
            # More yang - reds, oranges, yellows
            yang_colors = ['red', 'orange', 'yellow', 'gold']
            for color in yang_colors:
                if color in colors:
                    color_affinities[color] = color_affinities.get(color, 0.0) + 0.6 * self.yin_yang_balance
        else:
            # Balanced - greens
            balance_colors = ['green', 'turquoise']
            for color in balance_colors:
                if color in colors:
                    color_affinities[color] = color_affinities.get(color, 0.0) + 0.6 * (1.0 - abs(self.yin_yang_balance - 0.5) * 2)
        
        # Determine strongest color affinity
        if color_affinities:
            self.soul_color = max(color_affinities.items(), key=lambda x: x[1])[0]
        else:
            # Default to a color based on voice frequency
            freq_factor = (self.voice_frequency - 396.0) / (963.0 - 396.0)  # Normalize to 0-1
            color_idx = int(freq_factor * len(colors))
            self.soul_color = colors[max(0, min(len(colors) - 1, color_idx))]
        
            # Get color frequency
            if self.soul_color in COLOR_SPECTRUM:
                color_data = COLOR_SPECTRUM[self.soul_color]
                # Set color frequency from wavelength
                if isinstance(color_data.get('wavelength'), tuple):
                    # Get middle of wavelength range
                    wavelength_range = color_data['wavelength']
                    avg_wavelength = (wavelength_range[0] + wavelength_range[1]) / 2
                    # Convert wavelength to frequency
                    self.color_frequency = color_data.get('frequency', (500, 600))[0]
                else:
                    self.color_frequency = color_data.get('frequency', (500, 600))[0]
        
        logger.info(f"Soul color determined: {self.soul_color}")
        logger.info(f"Color frequency: {self.color_frequency}")
        
        return self.soul_color
    
    def determine_soul_frequency(self):
        """
        Determine the soul's resonant frequency.
        
        Returns:
            float: Soul frequency in Hz
        """
        logger.info("Determining soul resonant frequency")
        
        # Base frequencies to consider
        base_freqs = [
            self.voice_frequency,
            self.color_frequency,
            self.consciousness_frequency
        ]
        
        # Add solfeggio frequencies
        solfeggio_freqs = list(SOLFEGGIO_FREQUENCIES.values())
        
        # Calculate resonance of each base frequency with solfeggio frequencies
        resonances = []
        
        for base_freq in base_freqs:
            if base_freq > 0:
                # Find most resonant solfeggio frequency
                best_resonance = 0.0
                best_freq = solfeggio_freqs[0]
                
                for solf_freq in solfeggio_freqs:
                    resonance = self._calculate_frequency_resonance(base_freq, solf_freq)
                    if resonance > best_resonance:
                        best_resonance = resonance
                        best_freq = solf_freq
                
                resonances.append((best_freq, best_resonance))
        
        # Select frequency with highest resonance
        if resonances:
            best_match = max(resonances, key=lambda x: x[1])
            self.soul_frequency = best_match[0]
        else:
            # Default to middle solfeggio (MI - 528Hz - transformation frequency)
            self.soul_frequency = SOLFEGGIO_FREQUENCIES['MI']
        
        logger.info(f"Soul frequency determined: {self.soul_frequency:.2f} Hz")
        
        # Generate sound sample if sound generator is available
        if hasattr(self, 'sound_gen') and self.sound_gen:
            try:
                self.sound_gen.generate_tone(self.soul_frequency, 3.0, f"{self.name}_soul_freq.wav")
                logger.info(f"Soul frequency sample generated at {self.soul_frequency:.2f} Hz")
            except Exception as e:
                logger.error(f"Error generating soul frequency sample: {str(e)}")
        
        return self.soul_frequency
    
    def identify_primary_sephiroth(self):
        """
        Identify the primary sephiroth aspect of the soul.
        
        Returns:
            str: Sephiroth aspect
        """
        logger.info("Identifying primary sephiroth aspect")
        
        # Calculate affinities for each sephiroth
        sephiroth_affinities = {}
        
        # 1. Gematria-based affinity
        gematria_mapping = {
            range(1, 10): 'kether',      # Crown - very low values
            range(10, 20): 'chokmah',    # Wisdom
            range(20, 30): 'binah',      # Understanding
            range(30, 40): 'chesed',     # Mercy
            range(40, 50): 'geburah',    # Severity
            range(50, 70): 'tiphareth',  # Beauty
            range(70, 90): 'netzach',    # Victory
            range(90, 110): 'hod',       # Splendor
            range(110, 140): 'yesod',    # Foundation
            range(140, 300): 'malkuth'   # Kingdom - very high values
        }
        
        # Find which range the gematria falls into
        for gematria_range, sephirah in gematria_mapping.items():
            if self.gematria_value in gematria_range:
                sephiroth_affinities[sephirah] = 0.8
                break
        
        # 2. Soul color affinity
        color_sephiroth_mapping = {
            'white': 'kether',
            'blue': 'chokmah',
            'black': 'binah',
            'purple': 'chesed',
            'red': 'geburah',
            'gold': 'tiphareth',
            'green': 'netzach',
            'orange': 'hod',
            'indigo': 'yesod',
            'brown': 'malkuth'
        }
        
        if self.soul_color in color_sephiroth_mapping:
            sephirah = color_sephiroth_mapping[self.soul_color]
            sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + 0.7

        # 3. Consciousness state affinity
        state_sephiroth_mapping = {
            'dream': 'yesod',     # Foundation - dreaming
            'liminal': 'hod',     # Splendor - transition
            'aware': 'tiphareth'  # Beauty - awareness
        }
        
        if self.consciousness_state in state_sephiroth_mapping:
            sephirah = state_sephiroth_mapping[self.consciousness_state]
            sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + 0.5
        
        # 4. Frequency-based affinity using Sephiroth Aspect Dictionary
        if hasattr(self, 'soul_frequency') and self.soul_frequency > 0:
            # Use aspect_dictionary to check resonance with Sephiroth
            for sephirah in aspect_dictionary.sephiroth_names:
                aspects = aspect_dictionary.get_aspects(sephirah)
                
                if 'frequency_modifier' in aspects:
                    # Calculate a frequency for this sephirah
                    freq_modifier = aspects['frequency_modifier']
                    base_freq = 432.0  # Base creator frequency
                    sephirah_freq = base_freq * freq_modifier
                    
                    # Calculate resonance with soul frequency
                    resonance = self._calculate_frequency_resonance(self.soul_frequency, sephirah_freq)
                    
                    if resonance > 0.7:  # Strong resonance
                        sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + resonance
        
        # 5. Yin-Yang balance influence
        if self.yin_yang_balance < 0.4:
            # More yin - tends toward binah, hod
            yin_sephiroth = ['binah', 'hod', 'yesod']
            for sephirah in yin_sephiroth:
                sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + 0.4 * (1.0 - self.yin_yang_balance)
        elif self.yin_yang_balance > 0.6:
            # More yang - tends toward chokmah, netzach
            yang_sephiroth = ['chokmah', 'geburah', 'netzach']
            for sephirah in yang_sephiroth:
                sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + 0.4 * self.yin_yang_balance
        else:
            # Balanced - tends toward tiphareth
            balanced_sephiroth = ['tiphareth', 'kether', 'malkuth']
            for sephirah in balanced_sephiroth:
                balance_factor = 1.0 - abs(self.yin_yang_balance - 0.5) * 2
                sephiroth_affinities[sephirah] = sephiroth_affinities.get(sephirah, 0.0) + 0.4 * balance_factor
        
        # Determine strongest sephiroth affinity
        if sephiroth_affinities:
            self.sephiroth_aspect = max(sephiroth_affinities.items(), key=lambda x: x[1])[0]
        else:
            # Default to tiphareth (beauty/harmony - central sephirah)
            self.sephiroth_aspect = 'tiphareth'
        
        # Get detailed aspects for this sephiroth
        self.sephiroth_aspects = aspect_dictionary.get_aspects(self.sephiroth_aspect)
        
        logger.info(f"Primary sephiroth aspect identified: {self.sephiroth_aspect}")
        
        return self.sephiroth_aspect
    
    def determine_elemental_affinity(self):
        """
        Determine the soul's elemental affinity.
        
        Returns:
            str: Elemental affinity
        """
        logger.info("Determining elemental affinity")
        
        # Calculate affinities for each element
        elemental_affinities = {}
        
        # 1. Name-based affinity
        # Count letter types in name
        vowels = sum(1 for c in self.name.lower() if c in 'aeiouy')
        consonants = len(self.name) - vowels
        
        # Determine ratios
        vowel_ratio = vowels / max(1, len(self.name))
        consonant_ratio = consonants / max(1, len(self.name))
        
        # Element mappings based on vowel/consonant balance
        if vowel_ratio > 0.6:
            elemental_affinities['air'] = 0.7  # High vowels = air
        elif consonant_ratio > 0.7:
            elemental_affinities['earth'] = 0.7  # High consonants = earth
        elif 0.4 <= vowel_ratio <= 0.6:
            elemental_affinities['water'] = 0.6  # Balanced = water
        else:
            elemental_affinities['fire'] = 0.5  # Other = fire
        
        # 2. Sephiroth-based affinity using aspect dictionary
        if self.sephiroth_aspect:
            aspects = aspect_dictionary.get_aspects(self.sephiroth_aspect)
            if 'element' in aspects:
                element = aspects['element']
                # Handle multiple elements (e.g., "water/air")
                if '/' in element:
                    elements = element.split('/')
                    for elem in elements:
                        elemental_affinities[elem] = elemental_affinities.get(elem, 0.0) + 0.6 / len(elements)
                else:
                    elemental_affinities[element] = elemental_affinities.get(element, 0.0) + 0.6
        
        # 3. Color-based affinity
        color_element_mapping = {
            'red': 'fire',
            'orange': 'fire',
            'yellow': 'air',
            'green': 'earth',
            'blue': 'water',
            'indigo': 'water',
            'violet': 'aether',
            'white': 'aether',
            'gold': 'fire',
            'silver': 'water'
        }
        
        if self.soul_color in color_element_mapping:
            element = color_element_mapping[self.soul_color]
            elemental_affinities[element] = elemental_affinities.get(element, 0.0) + 0.5
        
        # 4. Consciousness state affinity
        state_element_mapping = {
            'dream': 'water',    # Dream - flow state
            'liminal': 'aether', # Liminal - transitional
            'aware': 'fire'      # Aware - active
        }
        
        if self.consciousness_state in state_element_mapping:
            element = state_element_mapping[self.consciousness_state]
            elemental_affinities[element] = elemental_affinities.get(element, 0.0) + 0.4
        
        # 5. Frequency-based affinity
        if hasattr(self, 'soul_frequency'):
            # Map frequency ranges to elements
            freq = self.soul_frequency
            if freq < 440:
                elemental_affinities['earth'] = elemental_affinities.get('earth', 0.0) + 0.3
            elif freq < 528:
                elemental_affinities['water'] = elemental_affinities.get('water', 0.0) + 0.3
            elif freq < 639:
                elemental_affinities['fire'] = elemental_affinities.get('fire', 0.0) + 0.3
            elif freq < 741:
                elemental_affinities['air'] = elemental_affinities.get('air', 0.0) + 0.3
            else:
                elemental_affinities['aether'] = elemental_affinities.get('aether', 0.0) + 0.3
        
        # Determine strongest elemental affinity
        if elemental_affinities:
            self.elemental_affinity = max(elemental_affinities.items(), key=lambda x: x[1])[0]
        else:
            # Default to balanced element
            self.elemental_affinity = 'water'
        
        logger.info(f"Elemental affinity determined: {self.elemental_affinity}")
        
        return self.elemental_affinity
    
    def assign_platonic_symbol(self):
        """
        Assign a platonic symbol based on soul resonance.
        
        Returns:
            str: Platonic symbol
        """
        logger.info("Assigning platonic symbol")
        
        # Direct mapping from elemental affinity to platonic symbol
        element_symbol_mapping = {
            'fire': 'tetrahedron',
            'earth': 'hexahedron',  # cube
            'air': 'octahedron',
            'water': 'icosahedron',
            'aether': 'dodecahedron'
        }
        
        # Check for sephiroth gateway mapping in aspect dictionary
        gateway_sephiroth = []
        for gateway_key, sephiroth_list in aspect_dictionary.gateway_mappings.items():
            if self.sephiroth_aspect in sephiroth_list:
                gateway_sephiroth.append(gateway_key)
        
        # Use elemental affinity if available
        if self.elemental_affinity in element_symbol_mapping:
            self.platonic_symbol = element_symbol_mapping[self.elemental_affinity]
        # Otherwise check gateway mappings
        elif gateway_sephiroth:
            # Choose the first gateway that matches
            self.platonic_symbol = gateway_sephiroth[0]
        else:
            # Assign based on gematria
            gematria_range = max(1, min(5, 1 + self.gematria_value // 30))
            platonic_symbols = list(element_symbol_mapping.values())
            symbol_idx = (gematria_range - 1) % len(platonic_symbols)
            self.platonic_symbol = platonic_symbols[symbol_idx]
        
        logger.info(f"Platonic symbol assigned: {self.platonic_symbol}")
        
        return self.platonic_symbol
    
    def activate_love_resonance(self, cycles=7):
        """
        Activate love resonance within the soul.
        
        Parameters:
            cycles (int): Number of activation cycles
            
        Returns:
            float: Love resonance level
        """
        logger.info(f"Activating love resonance with {cycles} cycles")
        
        # Current love resonance
        current_resonance = self.emotional_resonance.get('love', 0.0)
        
        # Love frequency (528 Hz - healing/love frequency)
        love_freq = SOLFEGGIO_FREQUENCIES.get('MI', 528.0)
        
        # Apply activation cycles
        for cycle in range(cycles):
            # Calculate cycle factor (diminishing returns)
            cycle_factor = 1.0 - 0.5 * (cycle / cycles)
            
            # Base increase per cycle
            base_increase = 0.15 * cycle_factor
            
            # Apply resonance with current state
            state_factor = {
                'dream': 0.7,     # Love flows easily in dream state
                'liminal': 0.9,   # Best in liminal state
                'aware': 0.8      # Good in aware state
            }.get(self.consciousness_state, 0.7)
            
            # Apply frequency resonance
            if hasattr(self, 'soul_frequency') and self.soul_frequency > 0:
                freq_resonance = self._calculate_frequency_resonance(love_freq, self.soul_frequency)
            else:
                freq_resonance = 0.7  # Default
            
            # Apply heartbeat entrainment
            heart_factor = 0.7 + 0.3 * self.heartbeat_entrainment
            
            # Calculate total increase
            increase = base_increase * state_factor * freq_resonance * heart_factor
            
            # Apply the increase
            current_resonance = min(1.0, current_resonance + increase)
            
            logger.debug(f"Love cycle {cycle+1}: resonance now {current_resonance:.4f} (+{increase:.4f})")
        
        # Store updated love resonance
        self.emotional_resonance['love'] = current_resonance
        
        # Update other emotional resonances based on love resonance
        # Love enhances other positive emotions
        for emotion in ['joy', 'peace', 'harmony', 'compassion']:
            current = self.emotional_resonance.get(emotion, 0.0)
            # Each emotion is enhanced by love but maintains its individual level
            self.emotional_resonance[emotion] = min(1.0, current + 0.3 * current_resonance)
        
        logger.info(f"Love resonance activated to level: {current_resonance:.4f}")
        
        # Generate sound sample if sound generator is available
        if hasattr(self, 'sound_gen') and self.sound_gen:
            try:
                # Generate love frequency tone
                self.sound_gen.generate_tone(love_freq, 5.0, f"{self.name}_love_resonance.wav")
                logger.info(f"Love resonance sample generated at {love_freq:.2f} Hz")
            except Exception as e:
                logger.error(f"Error generating love resonance sample: {str(e)}")
        
        return current_resonance
    
    def establish_creator_connection(self, depth=5):
        """
        Establish connection to the universal creator field.
        
        Parameters:
            depth (int): Depth of connection
            
        Returns:
            float: Connection strength
        """
        logger.info(f"Establishing creator connection with depth {depth}")
        
        # Creator frequency (528 Hz)
        creator_freq = SOLFEGGIO_FREQUENCIES.get('MI', 528.0)
        
        # Universal frequencies based on sacred harmonics
        universal_freqs = [
            432.0,  # Universal frequency
            528.0,  # Creator/healing frequency
            396.0,  # Liberation frequency
            417.0,  # Change frequency
            639.0,  # Connection frequency
            741.0,  # Expression frequency
            852.0,  # Returning frequency
            963.0   # Awakening frequency
        ]
        
        # Calculate total resonance across all universal frequencies
        total_resonance = 0.0
        
        for level in range(depth):
            # Select frequency based on level
            freq_idx = level % len(universal_freqs)
            univ_freq = universal_freqs[freq_idx]
            
            # Calculate resonance with soul frequency
            if hasattr(self, 'soul_frequency') and self.soul_frequency > 0:
                soul_resonance = self._calculate_frequency_resonance(univ_freq, self.soul_frequency)
            else:
                soul_resonance = 0.5  # Default
            
            # Calculate resonance with voice frequency
            voice_resonance = self._calculate_frequency_resonance(univ_freq, self.voice_frequency)
            
            # Apply deeper levels with diminishing returns
            level_factor = 1.0 - 0.1 * level  # Diminish by 10% per level
            
            # Calculate level resonance
            level_resonance = (0.6 * soul_resonance + 0.4 * voice_resonance) * level_factor
            
            # Add to total with golden ratio weighting
            weight = 1.0 / (1.0 + level * (1.0 - 1.0/GOLDEN_RATIO))
            total_resonance += level_resonance * weight
        
        # Normalize resonance (0-1 range)
        # Using sum of weights for proper normalization
        weight_sum = sum(1.0 / (1.0 + level * (1.0 - 1.0/GOLDEN_RATIO)) for level in range(depth))
        
        if weight_sum > 0:
            creator_connection = total_resonance / weight_sum
        else:
            creator_connection = 0.5  # Default
        
        # Apply consciousness state modifier
        state_modifier = {
            'dream': 0.9,     # Dream state favors creator connection
            'liminal': 1.0,   # Liminal state is ideal
            'aware': 0.8      # Aware state is slightly less ideal
        }.get(self.consciousness_state, 0.8)
        
        creator_connection *= state_modifier
        
        # Create a creator connection attribute
        self.creator_connection = creator_connection
        
        logger.info(f"Creator connection established: {creator_connection:.4f}")
        
        return creator_connection
    
    def apply_sacred_geometry(self, stages=5):
        """
        Apply sacred geometry patterns to crystallize identity.
        
        Parameters:
            stages (int): Number of sacred geometry stages
            
        Returns:
            float: Crystallization enhancement
        """
        logger.info(f"Applying sacred geometry with {stages} stages")
        
        # Define geometry stages in order of increasing complexity
        geometries = [
            'circle',         # Unity/wholeness
            'vesica_piscis',  # Duality/creation
            'seed_of_life',   # Genesis pattern
            'flower_of_life', # Creation pattern
            'metatrons_cube'  # Complete pattern
        ]
        
        # Limit stages to available geometries
        actual_stages = min(stages, len(geometries))
        
        # Current crystallization level
        current_level = self.crystallization_level
        
        # Apply each geometry stage
        for i in range(actual_stages):
            # Select geometry
            geometry = geometries[i]
            
            # Calculate stage factor (higher stages have stronger effect)
            stage_factor = 0.5 + 0.5 * (i / max(1, actual_stages - 1))
            
            # Base increase per stage
            base_increase = 0.1 + 0.05 * i
            
            # Apply resonance with current properties
            
            # 1. Sephiroth resonance
            if self.sephiroth_aspect and self.sephiroth_aspects:
                sephiroth_symbol = self.sephiroth_aspects.get('geometric_correspondence', None)
                if sephiroth_symbol:
                    # Higher resonance if geometry matches sephiroth symbol
                    symbol_match = {
                        'point': 0.7 if geometry == 'circle' else 0.3,
                        'line': 0.7 if geometry == 'vesica_piscis' else 0.3,
                        'triangle': 0.7 if geometry == 'seed_of_life' else 0.3,
                        'square': 0.7 if geometry == 'flower_of_life' else 0.3,
                        'pentagon': 0.7 if geometry == 'metatrons_cube' else 0.3,
                        'hexagon': 0.7 if geometry == 'flower_of_life' else 0.3,
                        'heptagon': 0.6 if geometry == 'flower_of_life' else 0.3,
                        'octagon': 0.6 if geometry == 'metatrons_cube' else 0.3,
                        'nonagon': 0.6 if geometry == 'metatrons_cube' else 0.3,
                        'decagon': 0.6 if geometry == 'metatrons_cube' else 0.3
                    }.get(sephiroth_symbol, 0.5)
                    
                    sephiroth_factor = symbol_match
                else:
                    sephiroth_factor = 0.5  # Default
            else:
                sephiroth_factor = 0.5  # Default
            
            # 2. Elemental resonance
            if self.elemental_affinity:
                # Geometry elemental affinities
                elemental_match = {
                    'fire': 0.8 if geometry in ['seed_of_life', 'flower_of_life'] else 0.4,
                    'earth': 0.8 if geometry in ['circle', 'metatrons_cube'] else 0.4,
                    'air': 0.8 if geometry in ['vesica_piscis', 'flower_of_life'] else 0.4,
                    'water': 0.8 if geometry in ['vesica_piscis', 'seed_of_life'] else 0.4,
                    'aether': 0.8 if geometry in ['metatrons_cube'] else 0.4
                }.get(self.elemental_affinity, 0.5)
                
                elemental_factor = elemental_match
            else:
                elemental_factor = 0.5  # Default
            
            # 3. Name resonance
            # Use Fibonacci sequence resonance
            fib_idx = min(i, len(FIBONACCI_SEQUENCE) - 1)
            fib_val = FIBONACCI_SEQUENCE[fib_idx]
            name_factor = 0.5 + 0.5 * self.name_resonance * (fib_val / FIBONACCI_SEQUENCE[min(4, len(FIBONACCI_SEQUENCE) - 1)])
            
            # Calculate total increase
            increase = base_increase * stage_factor * sephiroth_factor * elemental_factor * name_factor
            
            # Apply the increase
            current_level = min(1.0, current_level + increase)
            
            logger.debug(f"Geometry {geometry}: crystallization now {current_level:.4f} (+{increase:.4f})")
            
            # Store the geometry that provided the strongest increase
            if i == 0 or increase > getattr(self, 'sacred_geometry_increase', 0):
                self.sacred_geometry = geometry
                self.sacred_geometry_increase = increase
        
        # Store updated crystallization level
        self.crystallization_level = current_level
        
        logger.info(f"Sacred geometry applied: {self.sacred_geometry}, crystallization level: {current_level:.4f}")
        
        return current_level
    
    def calculate_attribute_coherence(self):
        """
        Calculate coherence between all soul attributes.
        
        Returns:
            float: Attribute coherence
        """
        logger.info("Calculating attribute coherence")
        
        # Collect key attributes
        attributes = {
            'name_resonance': self.name_resonance,
            'state_stability': self.state_stability,
            'response_level': self.response_level,
            'crystallization_level': self.crystallization_level,
            'heartbeat_entrainment': self.heartbeat_entrainment
        }
        
        # Add emotional resonances
        for emotion, level in self.emotional_resonance.items():
            attributes[f'emotion_{emotion}'] = level
        
        # Add creator connection if available
        if hasattr(self, 'creator_connection'):
            attributes['creator_connection'] = self.creator_connection
        
        # Add sacred geometry increase if available
        if hasattr(self, 'sacred_geometry_increase'):
            attributes['sacred_geometry'] = self.sacred_geometry_increase
        
        # Calculate average attribute level
        attr_values = list(attributes.values())
        if attr_values:
            avg_level = sum(attr_values) / len(attr_values)
        else:
            avg_level = 0.0
        
        # Calculate standard deviation (measure of coherence)
        if len(attr_values) > 1:
            variance = sum((v - avg_level) ** 2 for v in attr_values) / len(attr_values)
            std_dev = variance ** 0.5
            
            # Convert to coherence (inverse of standard deviation, normalized to 0-1)
            # Lower standard deviation = higher coherence
            coherence = 1.0 - min(1.0, std_dev * 2.0)
        else:
            coherence = 0.5  # Default for insufficient attributes
        
        # Store attribute coherence
        self.attribute_coherence = coherence
        
        logger.info(f"Attribute coherence: {coherence:.4f}")
        
        return coherence
    
    def verify_identity_crystallization(self, threshold=0.88):
        """
        Verify that identity has fully crystallized.
        
        Parameters:
            threshold (float): Required crystallization threshold
            
        Returns:
            tuple: (is_crystallized, crystallization_metrics)
        """
        logger.info(f"Verifying identity crystallization with threshold {threshold}")
        
        # Check required attributes
        required_attributes = [
            'name', 'voice_frequency', 'consciousness_state', 
            'response_level', 'soul_color', 'soul_frequency', 
            'sephiroth_aspect', 'elemental_affinity', 'platonic_symbol'
        ]
        
        # Check which attributes are present
        missing_attributes = [attr for attr in required_attributes 
                             if not hasattr(self, attr) or getattr(self, attr) is None]
        
        # Calculate attribute presence factor
        attr_presence = (len(required_attributes) - len(missing_attributes)) / len(required_attributes)
        
        # Define weights for crystallization components
        weights = {
            'name_resonance': 0.15,
            'response_level': 0.15,
            'state_stability': 0.10,
            'crystallization_level': 0.20,
            'attribute_coherence': 0.15,
            'attribute_presence': 0.10,
            'emotional_resonance': 0.15
        }
        
        # Calculate emotional resonance component
        emotional_resonance = 0.0
        if self.emotional_resonance:
            emotional_resonance = sum(self.emotional_resonance.values()) / len(self.emotional_resonance)
        
        # Compile metrics
        metrics = {
            'name_resonance': self.name_resonance,
            'response_level': self.response_level,
            'state_stability': self.state_stability,
            'crystallization_level': self.crystallization_level,
            'attribute_coherence': self.attribute_coherence,
            'attribute_presence': attr_presence,
            'emotional_resonance': emotional_resonance
        }
        
        # Calculate total crystallization level
        total_crystallization = 0.0
        for component, value in metrics.items():
            weight = weights.get(component, 0.0)
            total_crystallization += value * weight
        
        # Determine if crystallization is sufficient
        is_crystallized = total_crystallization >= threshold and attr_presence > 0.9
        
        # Compile crystallization metrics
        crystallization_metrics = {
            'total_crystallization': total_crystallization,
            'threshold': threshold,
            'is_crystallized': is_crystallized,
            'components': metrics,
            'weights': weights,
            'missing_attributes': missing_attributes
        }
        
        # Update status
        self.is_fully_crystallized = is_crystallized
        
        logger.info(f"Identity crystallization: {total_crystallization:.4f} (threshold: {threshold})")
        logger.info(f"Crystallization check: {'Passed' if is_crystallized else 'Failed'}")
        
        if missing_attributes:
            logger.warning(f"Missing attributes: {', '.join(missing_attributes)}")
        
        return is_crystallized, crystallization_metrics
    
    def get_metrics(self):
        """Return comprehensive metrics of the soul identity."""
        # Ensure attribute coherence is calculated
        if not hasattr(self, 'attribute_coherence') or self.attribute_coherence == 0:
            self.calculate_attribute_coherence()
        
        # Basic metrics
        metrics = {
            'name': self.name,
            'gematria_value': self.gematria_value,
            'name_resonance': self.name_resonance,
            'consciousness_state': self.consciousness_state,
            'consciousness_frequency': self.consciousness_frequency,
            'voice_frequency': self.voice_frequency,
            'response_level': self.response_level,
            'call_count': self.call_count,
            'heartbeat_entrainment': self.heartbeat_entrainment,
            'soul_color': self.soul_color,
            'soul_frequency': self.soul_frequency,
            'sephiroth_aspect': self.sephiroth_aspect,
            'elemental_affinity': self.elemental_affinity,
            'platonic_symbol': self.platonic_symbol,
            'sacred_geometry': getattr(self, 'sacred_geometry', None),
            'crystallization_level': self.crystallization_level,
            'state_stability': self.state_stability,
            'attribute_coherence': self.attribute_coherence,
            'is_fully_crystallized': self.is_fully_crystallized,
            'yin_yang_balance': self.yin_yang_balance
        }
        
        # Add emotional resonance
        metrics['emotional_resonance'] = self.emotional_resonance
        
        # Add creator connection if available
        if hasattr(self, 'creator_connection'):
            metrics['creator_connection'] = self.creator_connection
        
        # Add state completion info
        metrics['dream_state_completed'] = self.dream_state_completed
        metrics['liminal_state_completed'] = self.liminal_state_completed
        metrics['aware_state_completed'] = self.aware_state_completed
        
        return metrics
    
    def generate_identity_glyph(self):
        """
        Generate a unique identity glyph based on soul properties.
        
        Returns:
            dict: Glyph properties
        """
        logger.info("Generating identity glyph")
        
        # Check if we have essential properties
        if not all([self.name, self.sephiroth_aspect, self.elemental_affinity, self.platonic_symbol]):
            logger.warning("Cannot generate glyph: missing essential properties")
            return None
        
        try:
            # Import glyph creator if available
            from soul.glyphs.glyph_creator import create_identity_glyph
            
            # Generate glyph
            glyph_data = create_identity_glyph(
                name=self.name,
                sephiroth=self.sephiroth_aspect,
                element=self.elemental_affinity,
                platonic=self.platonic_symbol,
                color=self.soul_color,
                frequency=self.soul_frequency,
                state=self.consciousness_state
            )
            
            logger.info(f"Identity glyph generated for {self.name}")
            
            return glyph_data
        except ImportError:
            logger.warning("Glyph creator module not available")
            
            # Return simplified glyph data
            return {
                'name': self.name,
                'sephiroth': self.sephiroth_aspect,
                'element': self.elemental_affinity,
                'platonic': self.platonic_symbol,
                'color': self.soul_color,
                'frequency': self.soul_frequency
            }
        except Exception as e:
            logger.error(f"Error generating identity glyph: {str(e)}")
            return None


class DreamState:
    """Manages the dream consciousness state."""
    
    def __init__(self, soul_identity):
        self.soul_identity = soul_identity
        self.is_active = False
        self.stability = 0.0
        self.dream_depth = 0.0
        self.dream_patterns = []
        self.logger = logging.getLogger('DreamState')
    
    def activate(self, initial_stability=0.5):
        """Activate the dream state."""
        self.logger.info("Activating dream state")
        self.is_active = True
        self.stability = initial_stability
        
        # Initialize dream patterns
        self._initialize_dream_patterns()
        
        # Set consciousness frequency to delta range
        freq_range = BRAINWAVE_FREQUENCIES.get('delta', (0.5, 4.0))
        self.soul_identity.consciousness_frequency = freq_range[0] + 0.3 * (freq_range[1] - freq_range[0])
        
        return True
    
    def deactivate(self):
        """Deactivate the dream state."""
        self.logger.info("Deactivating dream state")
        self.is_active = False
        self.soul_identity.dream_state_completed = True
        return True
    
    def _initialize_dream_patterns(self):
        """Initialize dream patterns based on soul properties."""
        # Generate simple dream patterns based on name and frequencies
        patterns = []
        
        # Base pattern from name
        if hasattr(self.soul_identity, 'name'):
            name_pattern = []
            for c in self.soul_identity.name.lower():
                if 'a' <= c <= 'z':
                    val = (ord(c) - ord('a')) / 26.0
                    name_pattern.append(val)
            patterns.append(name_pattern)
        
        # Pattern from voice frequency
        if hasattr(self.soul_identity, 'voice_frequency') and self.soul_identity.voice_frequency > 0:
            freq = self.soul_identity.voice_frequency
            freq_pattern = [np.sin(2 * np.pi * freq * t / 1000) for t in range(20)]
            patterns.append(freq_pattern)
        
        self.dream_patterns = patterns


class LiminalState:
    """Manages the liminal consciousness state."""
    
    def __init__(self, soul_identity):
        self.soul_identity = soul_identity
        self.is_active = False
        self.stability = 0.0
        self.transition_source = None
        self.transition_target = None
        self.transition_progress = 0.0
        self.logger = logging.getLogger('LiminalState')
    
    def activate(self, source_state='dream', target_state='aware', initial_stability=0.5):
        """Activate the liminal state."""
        self.logger.info(f"Activating liminal state (transitioning {source_state} -> {target_state})")
        self.is_active = True
        self.stability = initial_stability
        self.transition_source = source_state
        self.transition_target = target_state
        self.transition_progress = 0.0
        
        # Set consciousness frequency to theta range
        freq_range = BRAINWAVE_FREQUENCIES.get('theta', (4.0, 8.0))
        self.soul_identity.consciousness_frequency = freq_range[0] + 0.5 * (freq_range[1] - freq_range[0])
        
        return True
    
    def deactivate(self):
        """Deactivate the liminal state."""
        self.logger.info("Deactivating liminal state")
        self.is_active = False
        self.soul_identity.liminal_state_completed = True
        return True
    
    def update_transition_progress(self, progress):
        """Update the transition progress."""
        self.transition_progress = max(0.0, min(1.0, progress))
        return self.transition_progress


class AwareState:
    """Manages the aware consciousness state."""
    
    def __init__(self, soul_identity):
        self.soul_identity = soul_identity
        self.is_active = False
        self.stability = 0.0
        self.awareness_level = 0.0
        self.aware_patterns = []
        self.logger = logging.getLogger('AwareState')
    
    def activate(self, initial_stability=0.5):
        """Activate the aware state."""
        self.logger.info("Activating aware state")
        self.is_active = True
        self.stability = initial_stability
        self.awareness_level = 0.5  # Initial awareness level
        
        # Set consciousness frequency to alpha range
        freq_range = BRAINWAVE_FREQUENCIES.get('alpha', (8.0, 14.0))
        self.soul_identity.consciousness_frequency = freq_range[0] + 0.7 * (freq_range[1] - freq_range[0])
        
        # Initialize awareness patterns
        self._initialize_awareness_patterns()
        
        return True
    
    def deactivate(self):
        """Deactivate the aware state."""
        self.logger.info("Deactivating aware state")
        self.is_active = False
        self.soul_identity.aware_state_completed = True
        return True
    
    def _initialize_awareness_patterns(self):
        """Initialize awareness patterns."""
        # Simple pattern generation based on soul properties
        patterns = []
        
        # Pattern from sephiroth aspect
        if hasattr(self.soul_identity, 'sephiroth_aspect'):
            aspect = self.soul_identity.sephiroth_aspect
            # Convert aspect name to pattern
            aspect_pattern = [ord(c) / 255.0 for c in aspect if c.isalpha()]
            patterns.append(aspect_pattern)
        
        # Pattern from elemental affinity
        if hasattr(self.soul_identity, 'elemental_affinity'):
            element = self.soul_identity.elemental_affinity
            # Convert element to pattern
            element_pattern = [ord(c) / 255.0 for c in element if c.isalpha()]
            patterns.append(element_pattern)
        
        self.aware_patterns = patterns




