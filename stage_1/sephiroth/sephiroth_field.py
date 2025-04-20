"""
Sephiroth Field Implementation

This module implements the Sephiroth dimensions - the spiritual realms
corresponding to the Tree of Life where souls evolve and refine their qualities.

Each Sephirah represents a different aspect of divinity/creation and imparts
specific qualities to souls that traverse through it. The field implementation
maintains unique properties for each Sephirah while sharing common mechanics.

Author: Soul Development Framework Team
"""

import numpy as np
import logging
import uuid
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base field system
from field_system import FieldSystem

# Import sacred geometry patterns
from shared.flower_of_life import FlowerOfLife
from shared.merkaba import Merkaba
from shared.seed_of_life import SeedOfLife
from shared.tree_of_life import TreeOfLife

# Import platonic solids
from platonics.tetrahedron import Tetrahedron
from platonics.hexahedron import Hexahedron
from platonics.octahedron import Octahedron
from platonics.icosahedron import Icosahedron
from platonics.dodecahedron import Dodecahedron

# Import Sephiroth aspects
from stage_1.sephiroth.sephiroth_aspect_dictionary import aspect_dictionary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sephiroth_field.log'
)
logger = logging.getLogger('sephiroth_field')

class SephirothField(FieldSystem):
    """
    Implementation of the Sephiroth dimensions.
    
    This class provides a unified implementation for all Sephiroth fields,
    with properties that vary based on the specific Sephirah. Each Sephirah
    represents a different divine quality and imparts those characteristics
    to souls that traverse through it.
    """
    
    def __init__(self, sephirah, dimensions=(64, 64, 64), edge_of_chaos_ratio=0.618, 
                creator_resonance=0.7, field_name=None):
        """
        Initialize a new Sephiroth field for a specific Sephirah.
        
        Args:
            sephirah (str): The specific Sephirah (e.g., "kether", "chokmah", etc.)
            dimensions (tuple): 3D dimensions of the field (x, y, z)
            edge_of_chaos_ratio (float): The edge of chaos parameter (default: golden ratio inverse)
            creator_resonance (float): Strength of the creator's resonance
            field_name (str): Optional custom name for the field
        """
        # Validate and normalize sephirah name
        self.sephirah = sephirah.lower()
        
        # Get valid Sephiroth from aspect dictionary
        valid_sephiroth = aspect_dictionary.sephiroth_names
        
        if self.sephirah not in valid_sephiroth:
            raise ValueError(f"Unknown Sephirah: {sephirah}. Valid options are: {', '.join(valid_sephiroth)}")
        
        # Generate field name if not provided
        if field_name is None:
            field_name = f"{self.sephirah.capitalize()} Field"
        
        # Get aspects for this Sephirah from the dictionary
        self.aspects = aspect_dictionary.get_aspects(self.sephirah)
        if not self.aspects:
            raise ValueError(f"Could not retrieve aspects for Sephirah: {self.sephirah}")
        
        # Get base frequency based on Sephirah
        base_frequency = self._get_sephirah_frequency()
        
        # Initialize base field
        super().__init__(dimensions=dimensions, field_name=field_name, 
                        edge_of_chaos_ratio=edge_of_chaos_ratio,
                        base_frequency=base_frequency)
        
        # Sephiroth-specific properties
        self.creator_resonance = creator_resonance
        self.divine_quality = self._get_divine_quality()
        self.stability_modifier = self._get_stability_modifier()
        self.resonance_multiplier = self._get_resonance_multiplier()
        self.dimensional_position = self._get_dimensional_position()
        self.platonic_solid = self._get_platonic_solid()
        
        # Track soul paths through this field
        self.soul_paths = []
        
        # Track divine qualities imparted to souls
        self.imparted_qualities = {}
        
        # Sacred patterns for this Sephirah
        self.sacred_patterns = {}
        self.sacred_solids = []
        
        # Initialize quantum field with Sephirah-specific properties
        self.initialize_quantum_field(base_amplitude=self._get_base_amplitude())
        
        # Embed sacred geometry patterns based on Sephirah
        self.embed_sacred_geometry()
        
        logger.info(f"Initialized {self.sephirah.capitalize()} field with base frequency {base_frequency} Hz")
    
    def _get_sephirah_frequency(self):
        """
        Get the base frequency for this Sephirah.
        
        Returns:
            float: Base frequency in Hz
        """
        # Use BASE_FREQUENCY from the aspect_dictionary (usually 432.0)
        base_creator_frequency = getattr(aspect_dictionary, 'BASE_FREQUENCY', 432.0)
        
        # Get the frequency modifier for this Sephirah
        if 'frequency_modifier' not in self.aspects:
            raise AttributeError(f"Frequency modifier not found in aspects for {self.sephirah}")
        
        # Apply the Sephirah's frequency modifier
        return base_creator_frequency * self.aspects['frequency_modifier']
    
    def _get_divine_quality(self):
        """
        Get the primary divine quality associated with this Sephirah.
        
        Returns:
            dict: Divine quality information
        """
        # Check required fields exist in aspects
        required_fields = ['name', 'title', 'primary_aspects', 'element', 'color']
        missing_fields = [field for field in required_fields if field not in self.aspects]
        
        if missing_fields:
            raise AttributeError(f"Missing required fields in aspects for {self.sephirah}: {', '.join(missing_fields)}")
        
        # Create a divine quality dict based on these aspects
        return {
            "name": self.aspects['title'],
            "description": f"The divine quality of {self.aspects['title']}",
            "primary_aspects": self.aspects['primary_aspects'],
            "secondary_aspects": self.aspects.get('secondary_aspects', []),
            "element": self.aspects['element'],
            "color": self.aspects['color'],
            "chakra": self.aspects.get('chakra_correspondence'),
            "strength": self.aspects.get('frequency_modifier', 1.0)
        }
    
    def _get_stability_modifier(self):
        """
        Get the stability modifier for this Sephirah.
        
        Higher values create more stable fields, lower values are more chaotic/dynamic.
        
        Returns:
            float: Stability modifier
        """
        # Use frequency_modifier from aspects (higher = more stable)
        if 'frequency_modifier' not in self.aspects:
            raise AttributeError(f"Frequency modifier not found in aspects for {self.sephirah}")
            
        # Higher frequency modifiers correspond to higher stability
        return self.aspects['frequency_modifier'] * 1.2
    
    def _get_resonance_multiplier(self):
        """
        Get the resonance multiplier for this Sephirah.
        
        This affects how strongly the field resonates with souls and other fields.
        
        Returns:
            float: Resonance multiplier
        """
        # Use phi_harmonic_count and harmonic_count from aspects
        if 'phi_harmonic_count' not in self.aspects or 'harmonic_count' not in self.aspects:
            raise AttributeError(f"Harmonic counts not found in aspects for {self.sephirah}")
            
        # Calculate resonance based on phi harmonics and total harmonics
        phi_factor = self.aspects['phi_harmonic_count'] / 7.0  # Normalize to 0-1 based on Kether's value
        harmonic_factor = self.aspects['harmonic_count'] / 12.0  # Normalize to 0-1 based on Kether's value
        
        # Combine factors, giving more weight to phi harmonics
        return 0.9 + (0.6 * phi_factor + 0.4 * harmonic_factor)
    
    def _get_dimensional_position(self):
        """
        Get the dimensional position of this Sephirah in the Tree of Life.
        
        Returns:
            dict: Dimensional position information
        """
        # Extract position info from aspects
        if 'position' not in self.aspects or 'pillar' not in self.aspects:
            raise AttributeError(f"Position information not found in aspects for {self.sephirah}")
        
        # Create position dictionary
        position = {
            "level": self.aspects['position'],
            "pillar": self.aspects['pillar']
        }
        
        # Add additional position attributes
        if self.aspects['position'] <= 3:
            position["supernal"] = True
        elif self.aspects['position'] <= 6:
            position["ethical"] = True
        else:
            position["astral"] = True
            
        # Calculate coordinates based on position and pillar
        y = 1.0 - (self.aspects['position'] - 1) / 10.0
        
        if self.aspects['pillar'] == "middle":
            x = 0.0
        elif self.aspects['pillar'] == "right":
            x = 1.0
        else:  # left pillar
            x = -1.0
            
        position["coordinates"] = (x, y, 0.0)
        
        return position
    
    def _get_platonic_solid(self):
        """
        Get the platonic solid associated with this Sephirah.
        
        These solids represent the geometric structure of each Sephirah's energy.
        
        Returns:
            dict: Platonic solid information
        """
        # Check for geometric correspondence in aspects
        if 'geometric_correspondence' not in self.aspects:
            raise AttributeError(f"Geometric correspondence not found in aspects for {self.sephirah}")
        
        # Get the geometric correspondence
        geom = self.aspects['geometric_correspondence']
        
        # Map to detailed platonic solid information
        platonic_map = {
            "point": {
                "name": "point",
                "vertices": 1,
                "faces": 0,
                "element": "quintessence",
                "dimension": 0
            },
            "line": {
                "name": "line",
                "vertices": 2,
                "faces": 0,
                "element": "fire",
                "dimension": 1
            },
            "triangle": {
                "name": "triangle",
                "vertices": 3,
                "faces": 1,
                "element": "water",
                "dimension": 2
            },
            "tetrahedron": {
                "name": "tetrahedron",
                "vertices": 4,
                "faces": 4,
                "element": "fire",
                "dimension": 3
            },
            "octahedron": {
                "name": "octahedron",
                "vertices": 6,
                "faces": 8,
                "element": "air",
                "dimension": 3
            },
            "hexahedron": {
                "name": "hexahedron",
                "vertices": 8,
                "faces": 6,
                "element": "earth",
                "dimension": 3
            },
            "icosahedron": {
                "name": "icosahedron",
                "vertices": 12,
                "faces": 20,
                "element": "water",
                "dimension": 3
            },
            "dodecahedron": {
                "name": "dodecahedron",
                "vertices": 20,
                "faces": 12,
                "element": "aether",
                "dimension": 3
            },
            "spheroid": {
                "name": "spheroid",
                "vertices": float('inf'),
                "faces": float('inf'),
                "element": "air",
                "dimension": 3
            },
            "cube": {
                "name": "cube",
                "vertices": 8,
                "faces": 6,
                "element": "earth",
                "dimension": 3
            }
        }
        
        # Return the detailed information or create basic info if not in mapping
        if geom in platonic_map:
            return platonic_map[geom]
        else:
            return {
                "name": geom,
                "vertices": 0,
                "faces": 0,
                "element": self.aspects.get('element', 'unknown'),
                "dimension": 3
            }
    
    def _get_base_amplitude(self):
        """
        Get the base amplitude for the quantum field of this Sephirah.
        
        Returns:
            float: Base amplitude
        """
        # Use phi_harmonic_count as a basis for amplitude
        if 'phi_harmonic_count' not in self.aspects:
            raise AttributeError(f"Phi harmonic count not found in aspects for {self.sephirah}")
            
        # Scale to a reasonable amplitude range (0.5-1.0)
        return 0.5 + 0.5 * (self.aspects['phi_harmonic_count'] / 7.0)
    
    def embed_sacred_geometry(self, center_position=None):
        """
        Embed sacred geometry patterns into the Sephiroth field.
        
        These patterns create the template for soul transformation within this Sephirah.
        
        Args:
            center_position (tuple): Center position for the patterns (default: field center)
            
        Returns:
            dict: Dictionary of embedded pattern information
        """
        if center_position is None:
            # Default to center of field
            center_position = (self.dimensions[0]//2, self.dimensions[1]//2, self.dimensions[2]//2)
        
        # Determine which sacred geometry patterns to embed based on Sephirah
        patterns = {}
        
        # Tree of Life is embedded in all Sephiroth fields
        tree = TreeOfLife()
        tree_pattern = tree.generate_pattern(dimensions=self.dimensions, 
                                            highlight_sephirah=self.sephirah)
        patterns["tree_of_life"] = tree_pattern
        self.embed_pattern(tree_pattern, strength=0.7 * self.resonance_multiplier)
        
        # Embed Flower of Life in all Sephiroth with varying strength
        flower = FlowerOfLife()
        flower_pattern = flower.generate_pattern(dimensions=self.dimensions, 
                                               center_position=center_position)
        patterns["flower_of_life"] = flower_pattern
        self.embed_pattern(flower_pattern, strength=0.5 * self.resonance_multiplier)
        
        # Seed of Life has different significance for different Sephiroth
        seed = SeedOfLife()
        seed_pattern = seed.generate_pattern(dimensions=self.dimensions,
                                           center_position=center_position)
        patterns["seed_of_life"] = seed_pattern
        
        # Strength varies by Sephirah's position in the Tree
        seed_strength = 0.4 * self.resonance_multiplier
        if 'position' in self.aspects:
            # Higher Sephiroth have stronger seed of life connection
            seed_strength *= (11 - self.aspects['position']) / 10.0
            
        self.embed_pattern(seed_pattern, strength=seed_strength)
        
        # Merkaba represents higher dimensional movement
        # More significant for certain Sephiroth
        merkaba = Merkaba()
        merkaba_pattern = merkaba.generate_pattern(dimensions=self.dimensions,
                                                center_position=center_position)
        patterns["merkaba"] = merkaba_pattern
        
        # Merkaba strength based on Sephirah properties
        merkaba_strength = 0.3 * self.resonance_multiplier
        
        # Higher spiritual connection in certain Sephiroth
        if self.aspects.get('pillar') == 'middle':
            merkaba_strength *= 1.5  # Stronger in middle pillar
        
        self.embed_pattern(merkaba_pattern, strength=merkaba_strength)
        
        # Store patterns for future reference
        self.sacred_patterns = patterns
        
        # Also embed the corresponding platonic solid for this Sephirah
        if 'geometric_correspondence' in self.aspects:
            self.embed_platonic_solid(self.aspects['geometric_correspondence'])
        
        logger.info(f"Embedded sacred geometry patterns in {self.sephirah} field")
        
        return patterns
    
    def calculate_resonance(self, frequency):
        """
        Calculate how strongly a frequency resonates with this Sephirah.
        
        Args:
            frequency (float): Frequency to check for resonance
            
        Returns:
            float: Resonance strength (0.0-1.0)
        """
        if not frequency or frequency <= 0:
            return 0.0
            
        # Get the base frequency for this Sephirah
        sephirah_freq = self._get_sephirah_frequency()
        
        # Exact match
        if abs(frequency - sephirah_freq) < 0.1:
            return 1.0
            
        # Calculate ratio (ensure larger value is in numerator)
        if frequency > sephirah_freq:
            ratio = frequency / sephirah_freq
        else:
            ratio = sephirah_freq / frequency
            
        # Strongest resonance at simple ratios (1:1, 2:1, 3:2, etc.)
        # Using GOLDEN_RATIO from aspect_dictionary
        golden_ratio = getattr(aspect_dictionary, 'GOLDEN_RATIO', 1.618033988749895)
        harmonic_ratios = [1.0, 2.0, 3.0/2.0, 4.0/3.0, golden_ratio]
        
        # Find distance to closest harmonic ratio
        min_distance = min(abs(ratio - hr) for hr in harmonic_ratios)
        
        # Convert distance to resonance (closer = higher resonance)
        resonance = 1.0 / (1.0 + 4.0 * min_distance)
        
        # Apply Sephirah's resonance multiplier
        resonance *= self.resonance_multiplier
        
        # Apply harmonic principles (octaves resonate strongly)
        if abs(np.log2(frequency / sephirah_freq) % 1) < 0.1:
            resonance = min(1.0, resonance * 1.5)
            
        return resonance
    
    def embed_platonic_solid(self, solid_type=None, center_position=None, scale=1.0):
        """
        Embed a platonic solid pattern into the field using existing platonic classes.
        
        Args:
            solid_type (str): Type of platonic solid to embed (default: from aspects)
            center_position (tuple): Center position for the solid (default: field center)
            scale (float): Size scale factor
            
        Returns:
            dict: Information about the embedded solid
        """
        # Use default from aspects if not specified
        if solid_type is None:
            if hasattr(self, 'platonic_solid') and 'name' in self.platonic_solid:
                solid_type = self.platonic_solid['name']
            else:
                solid_type = self.aspects.get('geometric_correspondence', 'tetrahedron')
        
        # Use center of field if not specified
        if center_position is None:
            center_position = tuple(d // 2 for d in self.dimensions)
        
        # Create appropriate platonic solid based on type
        platonic = None
        element = ""
        
        if solid_type == 'tetrahedron':
            platonic = Tetrahedron()
            element = "fire"
        elif solid_type in ['hexahedron', 'cube']:
            platonic = Hexahedron()
            element = "earth"
        elif solid_type == 'octahedron':
            platonic = Octahedron()
            element = "air"
        elif solid_type == 'icosahedron':
            platonic = Icosahedron()
            element = "water"
        elif solid_type == 'dodecahedron':
            platonic = Dodecahedron()
            element = "aether"
        else:
            # For other shapes or unrecognized types, use a default
            logger.warning(f"Using default tetrahedron for unrecognized type '{solid_type}'")
            platonic = Tetrahedron()
            element = "fire"
        
        # Generate the pattern with the platonic solid class
        if platonic:
            pattern = platonic.generate_pattern(
                dimensions=self.dimensions,
                center_position=center_position,
                scale=scale
            )
            
            # Embed the pattern in the field
            strength = 0.6 * self.resonance_multiplier
            self.embed_pattern(pattern, strength=strength)
            
            # Record the platonic information
            solid_info = {
                "type": solid_type,
                "element": element,
                "center": center_position,
                "scale": scale,
                "strength": strength
            }
            
            self.sacred_solids.append(solid_info)
            
            logger.info(f"Embedded {solid_type} in {self.sephirah} field")
            
            return solid_info
        
        return None
    
    def transform_soul(self, soul, duration=1.0):
        """
        Transform a soul that traverses through this Sephiroth field.
        
        The soul acquires aspects from this Sephirah and is transformed
        according to the divine qualities present.
        
        Args:
            soul: The soul object to transform
            duration (float): Duration of exposure to this field
            
        Returns:
            dict: Transformation results
        """
        if not hasattr(soul, 'aspects'):
            raise AttributeError("Soul object must have an 'aspects' attribute for transformation")
            
        logger.info(f"Transforming soul in {self.sephirah} field for {duration} time units")
        
        # Record initial state
        initial_state = self._get_soul_state(soul)
        
        # Calculate resonance between soul and this Sephirah
        resonance = self._calculate_soul_resonance(soul)
        
        # Calculate transformation strength
        strength = resonance * min(1.0, duration / 5.0)
        
        # Apply transformation to soul
        transformation = self._apply_transformation(soul, strength)
        
        # Record final state
        final_state = self._get_soul_state(soul)
        
        # Record path through this field
        path_record = {
            "soul_id": getattr(soul, 'id', str(uuid.uuid4())),
            "entry_time": datetime.now().isoformat(),
            "duration": duration,
            "resonance": resonance,
            "transformation_strength": strength,
            "aspects_gained": transformation.get("aspects_gained", []),
            "initial_state": initial_state,
            "final_state": final_state
        }
        
        self.soul_paths.append(path_record)
        
        logger.info(f"Soul transformation complete with resonance {resonance:.4f}")
        
        return {
            "resonance": resonance,
            "strength": strength,
            "aspects_gained": transformation.get("aspects_gained", []),
            "transformation": transformation
        }
    
    def _get_soul_state(self, soul):
        """Get a snapshot of the soul's current state."""
        # Extract relevant properties for record-keeping
        state = {}
        
        if hasattr(soul, 'aspects'):
            state['aspects'] = soul.aspects.copy() if isinstance(soul.aspects, dict) else {}
            
        if hasattr(soul, 'frequency'):
            state['frequency'] = soul.frequency
            
        if hasattr(soul, 'resonance'):
            state['resonance'] = soul.resonance
            
        if hasattr(soul, 'consciousness_state'):
            state['consciousness_state'] = soul.consciousness_state
            
        return state
    
    def _calculate_soul_resonance(self, soul):
        """Calculate resonance between the soul and this Sephirah."""
        resonance = 0.5  # Base resonance
        
        # Frequency resonance (if soul has a frequency)
        if hasattr(soul, 'frequency') and soul.frequency > 0:
            freq_resonance = self.calculate_resonance(soul.frequency)
            resonance = 0.3 + 0.7 * freq_resonance
            
        # Aspect resonance (if soul has aspects matching this Sephirah)
        if hasattr(soul, 'aspects') and isinstance(soul.aspects, dict):
            # Get primary aspects for this Sephirah
            sephirah_aspects = self.aspects.get('primary_aspects', [])
            
            # Count matching aspects
            matching_aspects = sum(1 for aspect in sephirah_aspects 
                                 if aspect in soul.aspects)
            
            if sephirah_aspects:
                aspect_resonance = matching_aspects / len(sephirah_aspects)
                
                # Update overall resonance
                resonance = 0.4 * resonance + 0.6 * aspect_resonance
                
        return min(1.0, resonance)
    
    def _apply_transformation(self, soul, strength):
        """Apply transformation to the soul based on this Sephirah's qualities."""
        transformation = {
            "aspects_gained": [],
            "frequency_shift": 0.0,
            "consciousness_change": 0.0
        }
        
        # Ensure soul has an aspects dictionary
        if not hasattr(soul, 'aspects'):
            soul.aspects = {}
        elif not isinstance(soul.aspects, dict):
            soul.aspects = {}
            
        # Transfer primary aspects based on strength
        primary_aspects = self.aspects.get('primary_aspects', [])
        
        # Determine how many aspects to transfer based on strength
        num_aspects = max(1, int(len(primary_aspects) * strength))
        
        # Select the most important aspects first
        for i in range(min(num_aspects, len(primary_aspects))):
            aspect = primary_aspects[i]
            
            # Add aspect to soul if not already present
            if aspect not in soul.aspects:
                soul.aspects[aspect] = strength
                transformation["aspects_gained"].append(aspect)
            else:
                # Strengthen existing aspect
                soul.aspects[aspect] = min(1.0, soul.aspects[aspect] + strength * 0.3)
                
        # Adjust soul frequency toward this Sephirah's frequency
        if hasattr(soul, 'frequency') and soul.frequency > 0:
            sephirah_freq = self._get_sephirah_frequency()
            
            # Calculate frequency shift
            freq_shift = (sephirah_freq - soul.frequency) * strength * 0.3
            
            # Apply shift
            soul.frequency += freq_shift
            transformation["frequency_shift"] = freq_shift
            
        # Influence consciousness state if applicable
        if hasattr(soul, 'consciousness_state') and hasattr(soul, 'consciousness_frequency'):
            # Certain Sephiroth have stronger influence on consciousness
            consciousness_influence = {
                "kether": "aware",
                "yesod": "dream",
                "tiphareth": "aware",
                "hod": "liminal",
                "binah": "liminal"
            }
            
            if self.sephirah in consciousness_influence:
                target_state = consciousness_influence[self.sephirah]
                
                # Record the influence
                transformation["consciousness_influence"] = {
                    "target_state": target_state,
                    "strength": strength * 0.4
                }
                
                # Note: Actual state change would typically be handled by the soul's own methods
                
        # Record this transformation in the Sephirah's records
        self.imparted_qualities[getattr(soul, 'id', str(uuid.uuid4()))] = {
            "time": datetime.now().isoformat(),
            "strength": strength,
            "aspects": transformation["aspects_gained"],
            "frequency_shift": transformation["frequency_shift"]
        }
        
        return transformation
    
    def visualize_field(self, slice_idx=None, show=True, save_path=None):
        """
        Visualize a slice of the Sephiroth field.
        
        Args:
            slice_idx (int): Index of the slice to visualize (default: middle of z-axis)
            show (bool): Whether to display the visualization
            save_path (str): Path to save the visualization image
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        import matplotlib.pyplot as plt
        
        # Use middle slice if not specified
        if slice_idx is None:
            slice_idx = self.dimensions[2] // 2
            
        # Get the field slice
        field_slice = self.field[:, :, slice_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get color map based on Sephirah
        cmap = self._get_sephirah_colormap()
        
        # Plot the field
        im = ax.imshow(field_slice.T, cmap=cmap, origin='lower', 
                     interpolation='bilinear')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Field Potential')
        
        # Add title with Sephirah information
        title = f"{self.aspects['name']} ({self.aspects['title']}) Field"
        ax.set_title(title)
        
        # Add sephirah position indicator
        if 'coordinates' in self.dimensional_position:
            pos = self.dimensional_position['coordinates']
            # Scale to field dimensions
            x = int((pos[0] + 1) * self.dimensions[0] / 2)
            y = int(pos[1] * self.dimensions[1])
            
            # Ensure within bounds
            x = max(0, min(x, self.dimensions[0]-1))
            y = max(0, min(y, self.dimensions[1]-1))
            
            # Mark position
            ax.plot(x, y, 'o', markersize=10, color='white')
            
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved field visualization to {save_path}")
            
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def _get_sephirah_colormap(self):
        """Get a colormap appropriate for this Sephirah."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Default colormap
        default_cmap = 'viridis'
        
        # Try to use color from aspects
        if 'color' not in self.aspects:
            return default_cmap
            
        # Get the color
        color = self.aspects['color']
        
        # Map common color names to matplotlib colormaps
        cmap_mapping = {
            'white': 'gray',
            'grey': 'gray',
            'black': 'gray_r',
            'blue': 'Blues',
            'red': 'Reds',
            'gold': 'YlOrBr',
            'yellow': 'YlOrBr',
            'green': 'Greens',
            'orange': 'Oranges',
            'purple': 'Purples',
            'brown': 'YlOrBr_r'
        }
        
        # Find the best match if color is a string
        if isinstance(color, str):
            # Check for exact match
            for color_name, cmap in cmap_mapping.items():
                if color_name in color.lower():
                    return cmap
                    
        # Default fallback
        return default_cmap

    def get_aspect_relationships(self):
        """
        Get the relationships between this Sephirah and others.
        
        Returns:
            dict: Dictionary of relationships with other Sephiroth
        """
        return aspect_dictionary.get_all_relationships(self.sephirah)
    
    def get_relationship(self, other_sephirah):
        """
        Get the relationship between this Sephirah and another.
        
        Args:
            other_sephirah (str): Name of the other Sephirah
            
        Returns:
            dict: Relationship information or empty dict if no relationship
        """
        return aspect_dictionary.get_relationship(self.sephirah, other_sephirah)
    
    def create_gateway_to(self, other_sephirah, gateway_key=None):
        """
        Create a gateway between this Sephirah and another.
        
        Args:
            other_sephirah (str): The target Sephirah
            gateway_key (str): Platonic solid to use as gateway key (optional)
            
        Returns:
            dict: Gateway information
        """
        # Check if there's a relationship with the other Sephirah
        relationship = self.get_relationship(other_sephirah)
        
        if not relationship:
            logger.warning(f"No known relationship between {self.sephirah} and {other_sephirah}")
            # We can still create a gateway but it will be weaker
        
        # Determine the appropriate gateway key if not specified
        if gateway_key is None:
            # Check if there's a gateway mapping for this pair
            gateways = aspect_dictionary.gateway_mappings
            
            for key, sephiroth in gateways.items():
                if self.sephirah in sephiroth and other_sephirah in sephiroth:
                    gateway_key = key
                    break
            
            # Default if no mapping found
            if gateway_key is None:
                gateway_key = "tetrahedron"  # Simplest gateway
        
        # Create gateway properties
        gateway = {
            "from_sephirah": self.sephirah,
            "to_sephirah": other_sephirah,
            "key": gateway_key,
            "strength": relationship.get("strength", 0.5) if relationship else 0.5,
            "quality": relationship.get("quality", "connection") if relationship else "connection",
            "path_name": relationship.get("path_name", "") if relationship else "",
            "created": datetime.now().isoformat()
        }
        
        # If we have relationship data, use it to enhance the gateway
        if relationship:
            gateway["tarot"] = relationship.get("tarot", "")
            
            # Stronger gateways for named paths
            if "path_name" in relationship:
                gateway["strength"] += 0.2
        
        logger.info(f"Created gateway from {self.sephirah} to {other_sephirah} using {gateway_key} key")
        
        return gateway
    
    def connect_to_creator(self, strength=None):
        """
        Create a connection to the Creator (Kether).
        
        Args:
            strength (float): Connection strength (optional)
            
        Returns:
            dict: Connection information
        """
        # Default strength based on position if not specified
        if strength is None:
            # Closer Sephiroth have stronger connection
            if 'position' in self.aspects:
                strength = 1.0 - ((self.aspects['position'] - 1) * 0.08)
                # Ensure reasonable range
                strength = max(0.4, min(strength, 0.95))
            else:
                strength = 0.7  # Default
        
        # Create connection properties
        connection = {
            "sephirah": self.sephirah,
            "creator_resonance": self.creator_resonance * strength,
            "connection_strength": strength,
            "frequency_alignment": self._calculate_frequency_alignment(),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Connected {self.sephirah} to Creator with strength {strength:.4f}")
        
        return connection
    
    def _calculate_frequency_alignment(self):
        """Calculate frequency alignment with Creator."""
        # Get this Sephirah's frequency
        sephirah_freq = self._get_sephirah_frequency()
        
        # Get Creator (Kether) frequency
        creator_freq = aspect_dictionary.BASE_FREQUENCY
        
        # Calculate ratio (should be close to a harmonic multiple if well-aligned)
        if creator_freq > 0 and sephirah_freq > 0:
            ratio = creator_freq / sephirah_freq
            
            # Find closest whole number fraction
            best_alignment = 0.0
            for n in range(1, 13):  # Check up to 12th harmonic
                for m in range(1, 13):
                    harmonic_ratio = n / m
                    alignment = 1.0 - abs(ratio - harmonic_ratio) / (harmonic_ratio)
                    
                    if alignment > best_alignment:
                        best_alignment = alignment
            
            return best_alignment
        
        return 0.5  # Default alignment
    
    def get_metrics(self):
        """
        Get comprehensive metrics about this Sephiroth field.
        
        Returns:
            dict: Field metrics
        """
        # Calculate field energy metrics
        energy = np.sum(self.field)
        volume = np.prod(self.dimensions)
        avg_energy = energy / volume
        max_energy = np.max(self.field)
        variation = np.std(self.field) / avg_energy if avg_energy > 0 else 0
        
        # Calculate pattern metrics
        pattern_count = len(self.sacred_patterns)
        solid_count = len(self.sacred_solids)
        
        # Soul interaction metrics
        souls_transformed = len(self.imparted_qualities)
        
        # Create metrics dictionary
        metrics = {
            "sephirah": self.sephirah,
            "name": self.aspects.get('name', ''),
            "title": self.aspects.get('title', ''),
            "position": self.aspects.get('position', 0),
            "pillar": self.aspects.get('pillar', ''),
            "frequency": self._get_sephirah_frequency(),
            "element": self.aspects.get('element', ''),
            "color": self.aspects.get('color', ''),
            "field_energy": {
                "total": float(energy),
                "average": float(avg_energy),
                "maximum": float(max_energy),
                "variation": float(variation)
            },
            "patterns": {
                "count": pattern_count,
                "types": list(self.sacred_patterns.keys())
            },
            "platonic_solids": {
                "count": solid_count,
                "types": [solid["type"] for solid in self.sacred_solids] if hasattr(self, 'sacred_solids') else []
            },
            "activity": {
                "souls_transformed": souls_transformed,
                "qualities_imparted": sum(len(q.get("aspects", [])) for q in self.imparted_qualities.values())
            },
            "connection": {
                "creator_resonance": self.creator_resonance,
                "stability": self.stability_modifier,
                "resonance_multiplier": self.resonance_multiplier
            }
        }
        
        return metrics

# Example usage if this module is run directly
if __name__ == "__main__":
    # Create a Tiphareth field as an example
    tiphareth = SephirothField("tiphareth")
    
    # Print some field properties
    print(f"Tiphareth field frequency: {tiphareth._get_sephirah_frequency():.2f} Hz")
    print(f"Divine quality: {tiphareth.divine_quality['name']}")
    print(f"Platonic solid: {tiphareth.platonic_solid['name']}")
    
    # Example: Get relationship with another Sephirah
    relationship = tiphareth.get_relationship("netzach")
    print(f"\nRelationship with Netzach: {relationship}")
    
    # Example: Create a gateway
    gateway = tiphareth.create_gateway_to("netzach")
    print(f"\nGateway: {gateway}")
    
    # Example: Visualize field
    try:
        tiphareth.visualize_field(save_path="tiphareth_field.png")
        print("\nVisualization saved to tiphareth_field.png")
    except ImportError:
        print("\nMatplotlib not available for visualization")


