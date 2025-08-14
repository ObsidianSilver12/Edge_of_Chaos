# --- START OF FILE field_harmonics.py ---

"""
Field Harmonics Module

Provides harmonic, geometric, sound and color properties for fields in the soul
development framework. Implements the transformative effects of sacred geometry,
sound, and color on souls passing through the fields.

Author: Soul Development Framework Team
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import math

# Configure logging
logger = logging.getLogger(__name__)

class FieldHarmonics:
    """
    Manages harmonics, geometric properties, sound and color attributes for 
    fields in the soul development framework.
    """
    
    # Sephiroth-specific frequencies
    SEPHIROTH_FREQUENCIES = {
        "kether": 963.0,
        "chokmah": 852.0, 
        "binah": 741.0,
        "chesed": 396.0,
        "geburah": 639.0,
        "tiphareth": 528.0,
        "netzach": 417.0,
        "hod": 396.0,
        "yesod": 369.0,
        "malkuth": 285.0,
        "daath": 999.0  # Hidden Sephirah
    }
    
    # Sephiroth-specific colors
    SEPHIROTH_COLORS = {
        "kether": "#FFFFFF",  # White
        "chokmah": "#808080",  # Grey
        "binah": "#000000",  # Black
        "chesed": "#0000FF",  # Blue
        "geburah": "#FF0000",  # Red
        "tiphareth": "#FFFF00",  # Yellow/Gold
        "netzach": "#00FF00",  # Green
        "hod": "#FFA500",  # Orange
        "yesod": "#800080",  # Purple
        "malkuth": "#654321",  # Brown
        "daath": "#C0C0C0"   # Silver
    }
    
    # Platonic solid geometric properties
    PLATONIC_GEOMETRY = {
        "tetrahedron": {
            "vertices": 4,
            "faces": 4,
            "edges": 6,
            "associated_element": "fire",
            "associated_sephirah": "geburah",
            "resonance_multiplier": 1.2
        },
        "hexahedron": {  # Cube
            "vertices": 8,
            "faces": 6,
            "edges": 12,
            "associated_element": "earth",
            "associated_sephirah": "malkuth",
            "resonance_multiplier": 1.0
        },
        "octahedron": {
            "vertices": 6,
            "faces": 8,
            "edges": 12,
            "associated_element": "air",
            "associated_sephirah": "tiphareth",
            "resonance_multiplier": 1.5
        },
        "dodecahedron": {
            "vertices": 20,
            "faces": 12,
            "edges": 30,
            "associated_element": "aether",
            "associated_sephirah": "kether",
            "resonance_multiplier": 1.8
        },
        "icosahedron": {
            "vertices": 12,
            "faces": 20,
            "edges": 30,
            "associated_element": "water",
            "associated_sephirah": "chesed",
            "resonance_multiplier": 1.3
        }
    }
    
    # Sephiroth-geometry natural associations
    SEPHIROTH_GEOMETRY = {
        "kether": "dodecahedron",
        "chokmah": "octahedron",
        "binah": "icosahedron",
        "chesed": "icosahedron",
        "geburah": "tetrahedron",
        "tiphareth": "octahedron",
        "netzach": "icosahedron",
        "hod": "octahedron",
        "yesod": "icosahedron",
        "malkuth": "hexahedron"
    }

    @staticmethod
    def generate_sound_file(field_type: str, sound_params: Dict[str, Any], 
                        event_type: str, identifier: str = "") -> Optional[str]:
        """
        Generate an actual sound file from parameters.
        
        Args:
            field_type: Type of field ('void', 'kether', etc.)
            sound_params: Dictionary of sound parameters
            event_type: Type of event ("transition", "burst", "resonance", etc.)
            identifier: Additional identifier (soul ID, location, etc.)
            
        Returns:
            Path to the generated sound file or None if generation failed
        """
        # Import sound_generator if needed
        try:
            from shared.sound.sound_generator import SoundGenerator
            sound_gen_available = True
            
            if sound_gen_available:
                # Create sound generator
                sound_gen = SoundGenerator(output_dir="output/sounds/field_interactions")
                
                # Calculate base frequency from field type
                base_freq = 432.0  # Default frequency
                if field_type in FieldHarmonics.SEPHIROTH_FREQUENCIES:
                    base_freq = FieldHarmonics.SEPHIROTH_FREQUENCIES[field_type]

                # Calculate ratio for harmonics (using golden ratio as default)
                ratio = (1 + math.sqrt(5)) / 2
                
                # Generate harmonic tone with correct parameter names
                sound = sound_gen.generate_harmonic_tone(
                    base_frequency=base_freq,
                    harmonics=[1.0, ratio],
                    amplitudes=[0.8, 0.4],
                    duration=3.0,
                    fade_in_out=0.5
                )
                
                # Save the sound
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"field_interaction_{field_type}_{event_type}_{timestamp}.wav"
                sound_file = sound_gen.save_sound(sound, filename)
                
                if sound_file:
                    logger.info(f"Generated sound file for light interaction: {sound_file}")
                    
        except Exception as sound_err:
            logger.error(f"Error generating light interaction sound: {sound_err}", exc_info=True)
   
    @staticmethod
    def get_sephirah_frequency(sephirah_name: str) -> float:
        """
        Get the base frequency for a Sephirah.
        
        Args:
            sephirah_name: Name of the Sephirah
            
        Returns:
            Base frequency in Hz
            
        Raises:
            ValueError: If sephirah_name is invalid
        """
        sephirah = sephirah_name.lower()
        if sephirah not in FieldHarmonics.SEPHIROTH_FREQUENCIES:
            raise ValueError(f"Unknown Sephirah: {sephirah_name}")
        
        return FieldHarmonics.SEPHIROTH_FREQUENCIES[sephirah]
    
    @staticmethod
    def get_sephirah_color(sephirah_name: str) -> str:
        """
        Get the color for a Sephirah.
        
        Args:
            sephirah_name: Name of the Sephirah
            
        Returns:
            Color as hex string
            
        Raises:
            ValueError: If sephirah_name is invalid
        """
        sephirah = sephirah_name.lower()
        if sephirah not in FieldHarmonics.SEPHIROTH_COLORS:
            raise ValueError(f"Unknown Sephirah: {sephirah_name}")
        
        return FieldHarmonics.SEPHIROTH_COLORS[sephirah]
    
    @staticmethod
    def get_sephirah_geometry(sephirah_name: str) -> str:
        """
        Get the associated Platonic solid for a Sephirah.
        
        Args:
            sephirah_name: Name of the Sephirah
            
        Returns:
            Name of the Platonic solid
            
        Raises:
            ValueError: If sephirah_name is invalid
        """
        sephirah = sephirah_name.lower()
        if sephirah not in FieldHarmonics.SEPHIROTH_GEOMETRY:
            raise ValueError(f"Unknown Sephirah: {sephirah_name}")
        
        return FieldHarmonics.SEPHIROTH_GEOMETRY[sephirah]
    
    @staticmethod
    def generate_harmonic_series(base_frequency: float, count: int = 7) -> List[float]:
        """
        Generate a harmonic series from a base frequency.
        
        Args:
            base_frequency: Base frequency in Hz
            count: Number of harmonics to generate
            
        Returns:
            List of harmonic frequencies
            
        Raises:
            ValueError: If parameters are invalid
        """
        if base_frequency <= 0:
            raise ValueError("Base frequency must be positive")
        
        if count <= 0:
            raise ValueError("Harmonic count must be positive")
        
        # Generate harmonics (integer multiples of the base frequency)
        return [base_frequency * (i + 1) for i in range(count)]
    
    @staticmethod
    def generate_overtone_series(base_frequency: float, count: int = 5) -> List[float]:
        """
        Generate an overtone series from a base frequency.
        
        Args:
            base_frequency: Base frequency in Hz
            count: Number of overtones to generate
            
        Returns:
            List of overtone frequencies
            
        Raises:
            ValueError: If parameters are invalid
        """
        if base_frequency <= 0:
            raise ValueError("Base frequency must be positive")
        
        if count <= 0:
            raise ValueError("Overtone count must be positive")
        
        # Generate overtones (based on the harmonic series but with specific focus on musical overtones)
        overtones = []
        for i in range(count):
            # Calculate musical overtones using just intonation principles
            if i == 0:
                overtones.append(base_frequency)  # Fundamental
            elif i == 1:
                overtones.append(base_frequency * 3/2)  # Perfect fifth
            elif i == 2:
                overtones.append(base_frequency * 5/4)  # Major third
            elif i == 3:
                overtones.append(base_frequency * 7/4)  # Harmonic seventh
            elif i == 4:
                overtones.append(base_frequency * 2)  # Octave
            else:
                overtones.append(base_frequency * (i + 1))  # Higher overtones
        
        return overtones
    
    @staticmethod
    def calculate_resonance(frequency1: float, frequency2: float) -> float:
        """
        Calculate resonance between two frequencies.
        
        Args:
            frequency1: First frequency in Hz
            frequency2: Second frequency in Hz
            
        Returns:
            Resonance value between 0.0 and 1.0
        """
        if frequency1 <= 0 or frequency2 <= 0:
            return 0.0
        
        # Calculate the frequency ratio
        ratio = frequency1 / frequency2 if frequency1 <= frequency2 else frequency2 / frequency1
        
        # Check if frequencies are in harmonic relationship (simple ratio)
        harmonic_resonance = 0.0
        
        # Check for simple harmonic ratios (octaves, fifths, etc.)
        for numerator in range(1, 8):
            for denominator in range(1, 8):
                harmonic_ratio = numerator / denominator
                if abs(ratio - harmonic_ratio) < 0.02:  # Close to a simple ratio
                    harmonic_resonance = 1.0 - (abs(ratio - harmonic_ratio) * 50)
                    harmonic_resonance *= (8 - numerator) / 7 * (8 - denominator) / 7  # Simpler ratios resonate more
                    break
        
        # Direct resonance based on closeness of frequencies
        direct_resonance = 1.0 - min(1.0, abs(frequency1 - frequency2) / max(frequency1, frequency2))
        
        # Combined resonance (favor harmonic relationships)
        resonance = 0.7 * harmonic_resonance + 0.3 * direct_resonance
        
        return max(0.0, min(1.0, resonance))
    
    @staticmethod
    def calculate_geometric_resonance(soul_frequency: float, geometry_name: str) -> float:
        """
        Calculate resonance between a soul frequency and a sacred geometry.
        
        Args:
            soul_frequency: Soul frequency in Hz
            geometry_name: Name of the Platonic solid
            
        Returns:
            Geometric resonance value between 0.0 and 2.0
        """
        if geometry_name not in FieldHarmonics.PLATONIC_GEOMETRY:
            return 0.0
        
        geometry = FieldHarmonics.PLATONIC_GEOMETRY[geometry_name]
        
        # Get associated Sephirah
        associated_sephirah = geometry["associated_sephirah"]
        sephirah_frequency = FieldHarmonics.get_sephirah_frequency(associated_sephirah)
        
        # Calculate base resonance with the Sephirah frequency
        base_resonance = FieldHarmonics.calculate_resonance(soul_frequency, sephirah_frequency)
        
        # Apply geometric multiplier
        geometric_resonance = base_resonance * geometry["resonance_multiplier"]
        
        # Apply Platonic properties based on vertices/faces/edges
        vertex_factor = 1.0 + (geometry["vertices"] / 20.0) * 0.5
        
        # Final resonance
        final_resonance = geometric_resonance * vertex_factor
        
        return max(0.0, min(2.0, final_resonance))  # Can go above 1.0 for strong resonance
    
    @staticmethod
    def generate_geometry_grid_modifier(geometry_name: str, grid_resolution: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate an energy grid modifier based on sacred geometry.
        
        Args:
            geometry_name: Name of the Platonic solid
            grid_resolution: (x, y, z) grid resolution
            
        Returns:
            Grid modifier array
        """
        if geometry_name not in FieldHarmonics.PLATONIC_GEOMETRY:
            # Default to uniform grid if geometry is unknown
            return np.ones(grid_resolution)
        
        # Create grid indices
        x, y, z = np.indices(grid_resolution)
        x_norm = x / grid_resolution[0]
        y_norm = y / grid_resolution[1]
        z_norm = z / grid_resolution[2]
        
        # Calculate center point
        center_x, center_y, center_z = 0.5, 0.5, 0.5
        
        # Base grid modifier
        modifier = np.ones(grid_resolution)
        
        # Apply geometric patterns based on Platonic solid type
        if geometry_name == "tetrahedron":
            # Tetrahedron pattern - strongest at vertices
            v1 = (0.5, 0.5, 1.0)  # Top vertex
            v2 = (0.0, 0.0, 0.0)  # Bottom vertex 1
            v3 = (1.0, 0.0, 0.0)  # Bottom vertex 2
            v4 = (0.5, 1.0, 0.0)  # Bottom vertex 3
            
            # Calculate distance to vertices
            dist_v1 = np.sqrt((x_norm - v1[0])**2 + (y_norm - v1[1])**2 + (z_norm - v1[2])**2)
            dist_v2 = np.sqrt((x_norm - v2[0])**2 + (y_norm - v2[1])**2 + (z_norm - v2[2])**2)
            dist_v3 = np.sqrt((x_norm - v3[0])**2 + (y_norm - v3[1])**2 + (z_norm - v3[2])**2)
            dist_v4 = np.sqrt((x_norm - v4[0])**2 + (y_norm - v4[1])**2 + (z_norm - v4[2])**2)
            
            # Minimum distance to any vertex
            min_dist = np.minimum(np.minimum(dist_v1, dist_v2), np.minimum(dist_v3, dist_v4))
            
            # Create energy pattern
            modifier = 1.0 + (1.0 - min_dist) * 0.5
            
        elif geometry_name == "hexahedron":  # Cube
            # Cube/Hexahedron - strongest at edges
            dist_center = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2 + (z_norm - center_z)**2)
            
            # Distance to nearest edge
            edge_dist_x = np.minimum(x_norm, 1.0 - x_norm)
            edge_dist_y = np.minimum(y_norm, 1.0 - y_norm)
            edge_dist_z = np.minimum(z_norm, 1.0 - z_norm)
            min_edge_dist = np.minimum(np.minimum(edge_dist_x, edge_dist_y), edge_dist_z)
            
            # Create energy pattern
            edge_effect = 1.0 - min_edge_dist
            center_effect = 1.0 - dist_center
            modifier = 1.0 + (0.7 * edge_effect + 0.3 * center_effect) * 0.5
            
        elif geometry_name == "octahedron":
            # Octahedron - dual of cube, strongest at vertices
            vertices = [
                (0.5, 0.5, 0.0),  # Bottom
                (0.5, 0.5, 1.0),  # Top
                (0.0, 0.5, 0.5),  # Left
                (1.0, 0.5, 0.5),  # Right
                (0.5, 0.0, 0.5),  # Front
                (0.5, 1.0, 0.5)   # Back
            ]
            
            # Calculate minimum distance to any vertex
            min_dist = np.ones(grid_resolution) * np.sqrt(3)  # Max possible distance
            for vertex in vertices:
                dist = np.sqrt((x_norm - vertex[0])**2 + (y_norm - vertex[1])**2 + (z_norm - vertex[2])**2)
                min_dist = np.minimum(min_dist, dist)
            
            # Create energy pattern - balanced
            modifier = 1.0 + (1.0 - min_dist) * 0.8
            
        elif geometry_name == "dodecahedron":
            # Dodecahedron - approximation of 12 pentagonal faces
            # Complex shape, use phi-based harmonics
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            # Phi-based distance function
            phi_dist = np.sin(phi * 2 * np.pi * x_norm) * \
                       np.sin(phi * 2 * np.pi * y_norm) * \
                       np.sin(phi * 2 * np.pi * z_norm)
            
            dist_center = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2 + (z_norm - center_z)**2)
            
            # Create energy pattern - strong phi resonance
            modifier = 1.0 + (0.5 + 0.5 * phi_dist) * (1.0 - dist_center) * 1.2
            
        elif geometry_name == "icosahedron":
            # Icosahedron - 20 triangular faces
            # Also complex, use spherical harmonic approximation
            vertices = []
            
            # Generate approximate icosahedron vertices
            for i in range(12):
                if i < 2:
                    # Top and bottom vertices
                    vertices.append((0.5, 0.5, i))
                else:
                    # Middle vertices in a regular pentagon pattern
                    angle = 2 * np.pi * (i - 2) / 10
                    z = 0.25 if i % 2 == 0 else 0.75
                    vertices.append((
                        0.5 + 0.5 * np.cos(angle),
                        0.5 + 0.5 * np.sin(angle),
                        z
                    ))
            
            # Calculate minimum distance to any vertex
            min_dist = np.ones(grid_resolution) * np.sqrt(3)  # Max possible distance
            for vertex in vertices:
                dist = np.sqrt((x_norm - vertex[0])**2 + (y_norm - vertex[1])**2 + (z_norm - vertex[2])**2)
                min_dist = np.minimum(min_dist, dist)
            
            # Create energy pattern - fluid, water-like
            dist_center = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2 + (z_norm - center_z)**2)
            water_pattern = 0.5 + 0.5 * np.sin(8 * np.pi * (
                0.4 * np.sin(3 * np.pi * x_norm) +
                0.4 * np.sin(3 * np.pi * y_norm) +
                0.2 * np.sin(3 * np.pi * z_norm)
            ))
            
            modifier = 1.0 + ((1.0 - min_dist) * 0.6 + water_pattern * 0.4) * 0.7
        
        # Ensure values stay in reasonable range
        return np.clip(modifier, 0.5, 2.0)
    
    @staticmethod
    def apply_soul_geometric_transformation(
        soul_data: Dict[str, Any], 
        geometry_name: str,
        transformation_strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply geometric transformation to a soul based on sacred geometry.
        
        Args:
            soul_data: Soul data dictionary
            geometry_name: Name of the Platonic solid
            transformation_strength: Strength of transformation (0.0-1.0)
            
        Returns:
            Updated soul data
        """
        if geometry_name not in FieldHarmonics.PLATONIC_GEOMETRY:
            return soul_data
        
        # Make a copy to avoid modifying the original
        transformed_soul = soul_data.copy()
        
        # Get geometry properties
        geometry = FieldHarmonics.PLATONIC_GEOMETRY[geometry_name]
        element = geometry["associated_element"]
        
        # Scale transformation_strength
        effect_strength = min(1.0, max(0.0, transformation_strength))
        
        # Apply element-based transformations
        if element == "fire":  # Tetrahedron
            # Fire increases energy, transformation, will
            transformed_soul.setdefault('aspects', {})
            
            # Update or create energy aspect
            if 'energy' in transformed_soul['aspects']:
                transformed_soul['aspects']['energy']['strength'] = min(
                    1.0, transformed_soul['aspects']['energy']['strength'] + 0.1 * effect_strength
                )
            else:
                transformed_soul['aspects']['energy'] = {'strength': 0.3 * effect_strength}
                
            # Update or create will aspect
            if 'will' in transformed_soul['aspects']:
                transformed_soul['aspects']['will']['strength'] = min(
                    1.0, transformed_soul['aspects']['will']['strength'] + 0.08 * effect_strength
                )
            else:
                transformed_soul['aspects']['will'] = {'strength': 0.25 * effect_strength}
                
            # Slightly increase frequency
            if 'frequency' in transformed_soul:
                transformed_soul['frequency'] *= (1.0 + 0.02 * effect_strength)
                
        elif element == "earth":  # Hexahedron/Cube
            # Earth increases stability, manifestation, grounding
            transformed_soul.setdefault('aspects', {})
            
            # Update or create stability aspect
            if 'stability' in transformed_soul['aspects']:
                transformed_soul['aspects']['stability']['strength'] = min(
                    1.0, transformed_soul['aspects']['stability']['strength'] + 0.12 * effect_strength
                )
            else:
                transformed_soul['aspects']['stability'] = {'strength': 0.35 * effect_strength}
                
            # Update or create manifestation aspect
            if 'manifestation' in transformed_soul['aspects']:
                transformed_soul['aspects']['manifestation']['strength'] = min(
                    1.0, transformed_soul['aspects']['manifestation']['strength'] + 0.1 * effect_strength
                )
            else:
                transformed_soul['aspects']['manifestation'] = {'strength': 0.3 * effect_strength}
                
            # Slightly decrease frequency for grounding
            if 'frequency' in transformed_soul:
                transformed_soul['frequency'] *= (1.0 - 0.01 * effect_strength)
                
        elif element == "air":  # Octahedron
            # Air increases intellect, communication, balance
            transformed_soul.setdefault('aspects', {})
            
            # Update or create intellect aspect
            if 'intellect' in transformed_soul['aspects']:
                transformed_soul['aspects']['intellect']['strength'] = min(
                    1.0, transformed_soul['aspects']['intellect']['strength'] + 0.09 * effect_strength
                )
            else:
                transformed_soul['aspects']['intellect'] = {'strength': 0.28 * effect_strength}
                
            # Update or create balance aspect
            if 'balance' in transformed_soul['aspects']:
                transformed_soul['aspects']['balance']['strength'] = min(
                    1.0, transformed_soul['aspects']['balance']['strength'] + 0.11 * effect_strength
                )
            else:
                transformed_soul['aspects']['balance'] = {'strength': 0.32 * effect_strength}
                
            # Minimal frequency change - balancing effect
            if 'frequency' in transformed_soul:
                target_freq = 639.0  # Balanced frequency
                current_freq = transformed_soul['frequency']
                transformed_soul['frequency'] += (target_freq - current_freq) * 0.05 * effect_strength
                
        elif element == "water":  # Icosahedron
            # Water increases fluidity, emotion, adaptability
            transformed_soul.setdefault('aspects', {})
            
            # Update or create emotion aspect
            if 'emotion' in transformed_soul['aspects']:
                transformed_soul['aspects']['emotion']['strength'] = min(
                    1.0, transformed_soul['aspects']['emotion']['strength'] + 0.11 * effect_strength
                )
            else:
                transformed_soul['aspects']['emotion'] = {'strength': 0.33 * effect_strength}
                
            # Update or create adaptability aspect
            if 'adaptability' in transformed_soul['aspects']:
                transformed_soul['aspects']['adaptability']['strength'] = min(
                    1.0, transformed_soul['aspects']['adaptability']['strength'] + 0.09 * effect_strength
                )
            else:
                transformed_soul['aspects']['adaptability'] = {'strength': 0.27 * effect_strength}
                
            # Gentle frequency harmonization
            if 'frequency' in transformed_soul:
                harmonics = FieldHarmonics.generate_harmonic_series(417.0, 3)  # Water-related harmonics
                closest_harmonic = min(harmonics, key=lambda h: abs(h - transformed_soul['frequency']))
                transformed_soul['frequency'] += (closest_harmonic - transformed_soul['frequency']) * 0.07 * effect_strength
                
        elif element == "aether":  # Dodecahedron
            # Aether increases spirituality, unity, transcendence
            transformed_soul.setdefault('aspects', {})
            
            # Update or create spirituality aspect
            if 'spirituality' in transformed_soul['aspects']:
                transformed_soul['aspects']['spirituality']['strength'] = min(
                    1.0, transformed_soul['aspects']['spirituality']['strength'] + 0.13 * effect_strength
                )
            else:
                transformed_soul['aspects']['spirituality'] = {'strength': 0.4 * effect_strength}
                
            # Update or create unity aspect
            if 'unity' in transformed_soul['aspects']:
                transformed_soul['aspects']['unity']['strength'] = min(
                    1.0, transformed_soul['aspects']['unity']['strength'] + 0.12 * effect_strength
                )
            else:
                transformed_soul['aspects']['unity'] = {'strength': 0.38 * effect_strength}
                
            # Significant frequency increase toward higher consciousness
            if 'frequency' in transformed_soul:
                transformed_soul['frequency'] *= (1.0 + 0.03 * effect_strength)
                # Ensure phi ratio relationship
                phi = (1 + np.sqrt(5)) / 2
                phi_resonance = transformed_soul['frequency'] * phi
                transformed_soul['frequency'] += (phi_resonance - transformed_soul['frequency']) * 0.1 * effect_strength
        
        # Record the transformation
        if 'geometric_transformations' not in transformed_soul:
            transformed_soul['geometric_transformations'] = []
            
        transformed_soul['geometric_transformations'].append({
            'geometry': geometry_name,
            'element': element,
            'strength': effect_strength,
            'timestamp': None  # Would be set with datetime.now().isoformat() in actual use
        })
        
        return transformed_soul
    
    @staticmethod
    def get_live_sound_parameters(
        field_type: str, 
        field_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate live sound parameters based on field state.
        
        Args:
            field_type: Type of field (e.g., "void", "sephiroth", "guff")
            field_properties: Dictionary of field properties
            
        Returns:
            Dictionary of sound parameters
        """
        sound_parameters = {
            'base_frequency': 432.0,  # Default
            'waveform': 'sine',  # sine, square, triangle, sawtooth
            'modulation': 0.0,
            'harmonics': [],
            'amplitude': 0.5,
            'attack': 0.1,
            'decay': 0.1,
            'sustain': 0.8,
            'release': 0.3,
            'reverb': 0.2,
            'filter_cutoff': 1000,
            'resonance': 0.5
        }
        
        # Base frequency from field if available
        if 'base_frequency' in field_properties:
            sound_parameters['base_frequency'] = field_properties['base_frequency']
        
        # Adjust based on field type
        if field_type == "void":
            sound_parameters['waveform'] = 'sine'
            sound_parameters['amplitude'] = 0.4
            sound_parameters['reverb'] = 0.8
            sound_parameters['attack'] = 0.3
            sound_parameters['release'] = 0.8
            sound_parameters['filter_cutoff'] = 800
        
        elif field_type == "sephiroth":
            # Get Sephirah specific parameters if available
            sephirah_name = field_properties.get('sephiroth_name', '').lower()
            if sephirah_name in FieldHarmonics.SEPHIROTH_FREQUENCIES:
                sound_parameters['base_frequency'] = FieldHarmonics.SEPHIROTH_FREQUENCIES[sephirah_name]
                
                # Generate appropriate harmonics
                sound_parameters['harmonics'] = FieldHarmonics.generate_harmonic_series(
                    sound_parameters['base_frequency'], 5
                )
                
                # Sephirah-specific sound qualities
                if sephirah_name == "kether":
                    sound_parameters['waveform'] = 'sine'
                    sound_parameters['reverb'] = 0.9
                    sound_parameters['filter_cutoff'] = 3000
                elif sephirah_name in ["chokmah", "binah"]:
                    sound_parameters['waveform'] = 'triangle'
                    sound_parameters['resonance'] = 0.7
                elif sephirah_name in ["chesed", "geburah"]:
                    sound_parameters['waveform'] = 'sawtooth'
                    sound_parameters['modulation'] = 0.3
                elif sephirah_name == "tiphareth":
                    sound_parameters['waveform'] = 'sine'
                    sound_parameters['attack'] = 0.05
                    sound_parameters['sustain'] = 0.9
                    sound_parameters['filter_cutoff'] = 2000
                elif sephirah_name in ["netzach", "hod"]:
                    sound_parameters['waveform'] = 'triangle'
                    sound_parameters['modulation'] = 0.2
                elif sephirah_name == "yesod":
                    sound_parameters['waveform'] = 'sine'
                    sound_parameters['reverb'] = 0.7
                    sound_parameters['filter_cutoff'] = 1200
                elif sephirah_name == "malkuth":
                    sound_parameters['waveform'] = 'square'
                    sound_parameters['attack'] = 0.2
                    sound_parameters['filter_cutoff'] = 800
                    
                # Apply field state modifiers
                if 'stability' in field_properties:
                    sound_parameters['sustain'] *= field_properties['stability']
                    
                if 'resonance' in field_properties:
                    sound_parameters['resonance'] = field_properties['resonance']
                    
                if 'coherence' in field_properties:
                    sound_parameters['modulation'] *= (1.0 - field_properties['coherence'])
        
        elif field_type == "guff":
            sound_parameters['waveform'] = 'sine'
            sound_parameters['reverb'] = 0.6
            sound_parameters['filter_cutoff'] = 1500
            sound_parameters['attack'] = 0.15
            
            # Adjust based on soul capacity if available
            if 'soul_capacity' in field_properties and 'souls_stored' in field_properties:
                capacity = field_properties['soul_capacity']
                stored = len(field_properties['souls_stored'])
                
                if capacity > 0:
                    fill_ratio = stored / capacity
                    sound_parameters['amplitude'] = 0.3 + 0.4 * fill_ratio
                    sound_parameters['resonance'] = 0.3 + 0.5 * fill_ratio
        
        # Apply general field state modifiers
        if 'active' in field_properties:
            if not field_properties['active']:
                sound_parameters['amplitude'] *= 0.2
        
        # Entity effects
        if 'entities' in field_properties:
            entity_count = len(field_properties['entities'])
            if entity_count > 0:
                # More entities = more complex sound
                sound_parameters['modulation'] += min(0.5, entity_count * 0.02)
                
                # Add subtle pitch variation based on entity count
                sound_parameters['base_frequency'] *= (1.0 + 0.001 * entity_count)
        
        return sound_parameters
    
    @staticmethod
    def generate_live_sound_visualization(
        sound_parameters: Dict[str, Any],
        field_type: str,
        field_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate visualization parameters for live sound.
        
        Args:
            sound_parameters: Sound parameters dictionary
            field_type: Type of field
            field_properties: Dictionary of field properties
            
        Returns:
            Dictionary of visualization parameters
        """
        visualization = {
            'primary_color': '#3366CC',  # Default blue
            'secondary_color': '#66CCFF',
            'background_color': '#000033',
            'waveform_style': 'smooth',  # smooth, angular, particle
            'intensity': 0.7,
            'speed': 1.0,
            'complexity': 0.5,
            'symmetry': True,
            'pulse_effect': 0.3,
            'glow_effect': 0.5,
            'size': 0.7
        }
        
        # Get color based on field type
        if field_type == "sephiroth":
            sephirah_name = field_properties.get('sephiroth_name', '').lower()
            if sephirah_name in FieldHarmonics.SEPHIROTH_COLORS:
                visualization['primary_color'] = FieldHarmonics.SEPHIROTH_COLORS[sephirah_name]
                
                # Generate complementary color for secondary
                primary_hex = FieldHarmonics.SEPHIROTH_COLORS[sephirah_name].lstrip('#')
                r = int(primary_hex[0:2], 16)
                g = int(primary_hex[2:4], 16)
                b = int(primary_hex[4:6], 16)
                
                # Simple complementary calculation
                secondary_r = (r + 128) % 256
                secondary_g = (g + 128) % 256
                secondary_b = (b + 128) % 256
                
                visualization['secondary_color'] = f"#{secondary_r:02x}{secondary_g:02x}{secondary_b:02x}"
                
                # Background should be darker version of primary
                background_r = max(0, r - 150)
                background_g = max(0, g - 150)
                background_b = max(0, b - 150)
                
                visualization['background_color'] = f"#{background_r:02x}{background_g:02x}{background_b:02x}"
        
        # Match visualization parameters to sound
        if sound_parameters['waveform'] == 'sine':
            visualization['waveform_style'] = 'smooth'
        elif sound_parameters['waveform'] == 'triangle':
            visualization['waveform_style'] = 'angular'
        elif sound_parameters['waveform'] == 'square':
            visualization['waveform_style'] = 'angular'
            visualization['complexity'] = 0.3
        elif sound_parameters['waveform'] == 'sawtooth':
            visualization['waveform_style'] = 'particle'
            visualization['complexity'] = 0.7
        
        # Intensity based on amplitude
        visualization['intensity'] = sound_parameters['amplitude'] * 1.2
        
        # Speed based on frequency
        base_freq = sound_parameters['base_frequency']
        if base_freq < 400:
            visualization['speed'] = 0.7
        elif base_freq > 800:
            visualization['speed'] = 1.3
        
        # Effects based on reverb and resonance
        visualization['glow_effect'] = sound_parameters['reverb'] * 1.5
        visualization['pulse_effect'] = sound_parameters['resonance'] * 0.8
        
        # Symmetry based on field coherence
        if 'coherence' in field_properties:
            visualization['symmetry'] = field_properties['coherence'] > 0.5
        
        # Size based on field dimensions if available
        if 'dimensions' in field_properties:
            dimensions = field_properties['dimensions']
            avg_dim = sum(dimensions) / 3.0
            visualization['size'] = min(1.0, avg_dim / 100.0)
        
        return visualization
    
    @staticmethod
    def get_sephiroth_harmonic_sound_map() -> Dict[str, Dict[str, Any]]:
        """
        Get a complete mapping of Sephiroth to sound properties.
        
        Returns:
            Dictionary mapping Sephiroth names to sound properties
        """
        sound_map = {}
        
        for sephirah in FieldHarmonics.SEPHIROTH_FREQUENCIES.keys():
            base_freq = FieldHarmonics.SEPHIROTH_FREQUENCIES[sephirah]
            harmonics = FieldHarmonics.generate_harmonic_series(base_freq, 7)
            overtones = FieldHarmonics.generate_overtone_series(base_freq, 5)
            
            # Select appropriate waveform
            waveform = 'sine'
            if sephirah in ["geburah", "malkuth"]:
                waveform = 'square'
            elif sephirah in ["chokmah", "binah", "netzach", "hod"]:
                waveform = 'triangle'
            elif sephirah in ["chesed"]:
                waveform = 'sawtooth'
            
            # Select appropriate envelope
            attack = 0.1
            decay = 0.1
            sustain = 0.8
            release = 0.3
            
            if sephirah in ["kether", "tiphareth"]:
                attack = 0.05
                sustain = 0.9
                release = 0.5
            elif sephirah in ["malkuth"]:
                attack = 0.2
                sustain = 0.7
                release = 0.2
            elif sephirah in ["yesod"]:
                attack = 0.15
                sustain = 0.8
                release = 0.4
            
            # Create sound properties
            sound_map[sephirah] = {
                'base_frequency': base_freq,
                'harmonics': harmonics,
                'overtones': overtones,
                'waveform': waveform,
                'color': FieldHarmonics.SEPHIROTH_COLORS[sephirah],
                'envelope': {
                    'attack': attack,
                    'decay': decay,
                    'sustain': sustain,
                    'release': release
                },
                'effects': {
                    'reverb': 0.5,
                    'filter_cutoff': 1000,
                    'resonance': 0.5
                }
            }
            
            # Adjust specific effects
            if sephirah == "kether":
                sound_map[sephirah]['effects']['reverb'] = 0.9
                sound_map[sephirah]['effects']['filter_cutoff'] = 3000
            elif sephirah == "tiphareth":
                sound_map[sephirah]['effects']['reverb'] = 0.7
                sound_map[sephirah]['effects']['filter_cutoff'] = 2000
            elif sephirah == "yesod":
                sound_map[sephirah]['effects']['reverb'] = 0.7
                sound_map[sephirah]['effects']['filter_cutoff'] = 1200
            elif sephirah == "malkuth":
                sound_map[sephirah]['effects']['reverb'] = 0.3
                sound_map[sephirah]['effects']['filter_cutoff'] = 800
        
        return sound_map
    
    @staticmethod
    def get_platonic_solid_sound_map() -> Dict[str, Dict[str, Any]]:
        """
        Get a complete mapping of Platonic solids to sound properties.
        
        Returns:
            Dictionary mapping Platonic solid names to sound properties
        """
        sound_map = {}
        
        for geometry, properties in FieldHarmonics.PLATONIC_GEOMETRY.items():
            element = properties['associated_element']
            sephirah = properties['associated_sephirah']
            
            # Get base frequency from associated Sephirah
            base_freq = FieldHarmonics.SEPHIROTH_FREQUENCIES.get(sephirah, 432.0)
            
            # Select appropriate waveform based on element
            waveform = 'sine'
            if element == "fire":
                waveform = 'sawtooth'
            elif element == "earth":
                waveform = 'square'
            elif element == "air":
                waveform = 'triangle'
            elif element == "water":
                waveform = 'sine'
            elif element == "aether":
                waveform = 'sine'
            
            # Create sound properties
            sound_map[geometry] = {
                'base_frequency': base_freq,
                'harmonics': FieldHarmonics.generate_harmonic_series(base_freq, 5),
                'waveform': waveform,
                'element': element,
                'associated_sephirah': sephirah,
                'envelope': {
                    'attack': 0.1,
                    'decay': 0.1,
                    'sustain': 0.8,
                    'release': 0.3
                },
                'effects': {
                    'reverb': 0.5,
                    'filter_cutoff': 1000,
                    'resonance': 0.5,
                    'modulation': 0.0
                }
            }
            
            # Adjust by element
            if element == "fire":  # Tetrahedron
                sound_map[geometry]['envelope']['attack'] = 0.05
                sound_map[geometry]['envelope']['sustain'] = 0.6
                sound_map[geometry]['effects']['filter_cutoff'] = 2000
                sound_map[geometry]['effects']['resonance'] = 0.7
            elif element == "earth":  # Hexahedron/Cube
                sound_map[geometry]['envelope']['attack'] = 0.2
                sound_map[geometry]['envelope']['sustain'] = 0.9
                sound_map[geometry]['effects']['filter_cutoff'] = 800
                sound_map[geometry]['effects']['resonance'] = 0.3
            elif element == "air":  # Octahedron
                sound_map[geometry]['envelope']['attack'] = 0.08
                sound_map[geometry]['envelope']['release'] = 0.5
                sound_map[geometry]['effects']['filter_cutoff'] = 1500
                sound_map[geometry]['effects']['modulation'] = 0.2
            elif element == "water":  # Icosahedron
                sound_map[geometry]['envelope']['attack'] = 0.15
                sound_map[geometry]['envelope']['release'] = 0.6
                sound_map[geometry]['effects']['reverb'] = 0.7
                sound_map[geometry]['effects']['filter_cutoff'] = 1200
            elif element == "aether":  # Dodecahedron
                sound_map[geometry]['envelope']['attack'] = 0.03
                sound_map[geometry]['envelope']['release'] = 0.8
                sound_map[geometry]['effects']['reverb'] = 0.9
                sound_map[geometry]['effects']['filter_cutoff'] = 3000
        
        return sound_map

# --- END OF FILE field_harmonics.py ---
