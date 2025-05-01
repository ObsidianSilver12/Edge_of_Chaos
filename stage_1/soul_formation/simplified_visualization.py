"""
Soul Development Framework - Simplified Visualization System

This module provides a single, straightforward interface for generating
visualizations at key points in the soul formation process.
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SimpleVisualizer:
    """
    A simplified visualization system for the Soul Development Framework.
    
    This class provides methods to generate visualizations at key points
    in the soul formation process without complex hooks or integrations.
    """
    
    def __init__(self, enabled=True, save_to_disk=True, show_images=False,
                 output_dir="output/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            enabled: Whether visualization is enabled
            save_to_disk: Whether to save visualizations to disk
            show_images: Whether to display visualizations (interactive mode)
            output_dir: Directory to save visualizations
        """
        self.enabled = enabled
        self.save_to_disk = save_to_disk
        self.show_images = show_images
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if self.save_to_disk:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Visualization output directory created: {output_dir}")
            except OSError as e:
                logger.error(f"Failed to create visualization directory: {e}")
                self.save_to_disk = False
        
        # Initialize visualizer components lazily
        self._soul_visualizer_class = None
        self._field_visualizer = None
        self._journey_visualizer = None
        self._soul_visualizers = {}  # Cache for soul-specific visualizers
        
        logger.info(f"Visualization system initialized (enabled={enabled})")
    
    def _get_soul_visualizer(self, soul):
        """
        Get or create a soul visualizer for the given soul.
        
        Args:
            soul: The SoulSpark instance
            
        Returns:
            EnhancedSoulVisualizer or None if not available
        """
        if not self.enabled:
            return None
            
        soul_id = getattr(soul, 'spark_id', str(id(soul)))
        
        # If already created for this soul, return cached instance
        if soul_id in self._soul_visualizers:
            return self._soul_visualizers[soul_id]
        
        # If the class is not loaded yet, try to load it
        if self._soul_visualizer_class is None:
            try:
                from stage_1.soul_formation.soul_visualization_enhanced import EnhancedSoulVisualizer
                self._soul_visualizer_class = EnhancedSoulVisualizer
                logger.info("Soul visualizer class loaded")
            except ImportError as e:
                logger.error(f"Failed to import soul visualizer: {e}")
                return None
        
        # Create a new instance for this soul
        if self._soul_visualizer_class:
            try:
                soul_viz = self._soul_visualizer_class(
                    soul, 
                    output_dir=os.path.join(self.output_dir, "souls", soul_id)
                )
                self._soul_visualizers[soul_id] = soul_viz
                return soul_viz
            except Exception as e:
                logger.error(f"Failed to create soul visualizer: {e}")
        
        return None
    
    def _get_field_visualizer(self):
        """Get the field visualizer, initializing if needed."""
        if not self.enabled or self._field_visualizer is not None:
            return self._field_visualizer
            
        try:
            from stage_1.fields.field_visualization import FieldVisualizer
            self._field_visualizer = FieldVisualizer(
                output_dir=os.path.join(self.output_dir, "fields")
            )
            logger.info("Field visualizer loaded")
            return self._field_visualizer
        except ImportError as e:
            logger.error(f"Failed to import field visualizer: {e}")
            return None
    
    def _get_journey_visualizer(self):
        """Get the journey visualizer, initializing if needed."""
        if not self.enabled or self._journey_visualizer is not None:
            return self._journey_visualizer
            
        try:
            from stage_1.soul_formation.soul_journey_visualization import SoulJourneyVisualizer
            self._journey_visualizer = SoulJourneyVisualizer(
                output_dir=os.path.join(self.output_dir, "journey")
            )
            logger.info("Journey visualizer loaded")
            return self._journey_visualizer
        except ImportError as e:
            logger.error(f"Failed to import journey visualizer: {e}")
            return None
    
    def visualize_field_state(self, field_controller, stage="general"):
        """
        Visualize the current state of the field system.
        
        Args:
            field_controller: The FieldController instance
            stage: Name of the current stage (for logging)
        """
        if not self.enabled:
            return
        
        field_viz = self._get_field_visualizer()
        if not field_viz:
            return
            
        try:
            # Create field dashboard visualization
            field_viz.create_field_dashboard(
                field_controller, 
                show=self.show_images, 
                save=self.save_to_disk
            )
            logger.info(f"Field state visualization complete ({stage})")
        except Exception as e:
            logger.error(f"Error visualizing field state: {e}")
    
    def visualize_guff_strengthening(self, soul, field_controller):
        """
        Visualize a soul after Guff strengthening.
        
        Args:
            soul: The SoulSpark instance
            field_controller: The FieldController instance
        """
        if not self.enabled:
            return
        
        # Get soul visualizer
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                # Visualize core structure
                soul_viz.visualize_core_structure(
                    show=self.show_images, 
                    save=self.save_to_disk
                )
                logger.info("Guff strengthening core visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing soul after Guff strengthening: {e}")
            
            try:
                # Visualize energy harmonics
                soul_viz.visualize_energy_harmony(
                    show=self.show_images, 
                    save=self.save_to_disk
                )
                logger.info("Guff strengthening energy visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing soul energy: {e}")
        
        # Also visualize soul in field
        field_viz = self._get_field_visualizer()
        if field_viz:
            try:
                # Get current field
                current_field_key = getattr(soul, 'current_field_key', 'guff')
                field = field_controller.get_field(current_field_key)
                if field:
                    field_viz.visualize_soul_field_interaction(
                        soul, field,
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Soul-field interaction visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing soul in field: {e}")
    
    def visualize_sephiroth_journey(self, soul):
        """
        Visualize a soul's journey through the Sephiroth.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
            
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Check if visualize_sephiroth_journey method exists
                if hasattr(journey_viz, 'visualize_sephiroth_journey'):
                    journey_viz.visualize_sephiroth_journey(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Sephiroth journey visualization complete")
                # Try visualize_journey_timeline as a fallback
                elif hasattr(journey_viz, 'visualize_journey_timeline'):
                    journey_viz.visualize_journey_timeline(
                        soul,
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Journey timeline visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing Sephiroth journey: {e}")
            
        # Also visualize the soul's aspects
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                soul_viz.visualize_aspects_map(
                    show=self.show_images, 
                    save=self.save_to_disk
                )
                logger.info("Soul aspects visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing soul aspects: {e}")
    
    def visualize_creator_entanglement(self, soul):
        """
        Visualize a soul's entanglement with the Creator.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
            
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Check if the method exists
                if hasattr(journey_viz, 'visualize_creator_entanglement'):
                    journey_viz.visualize_creator_entanglement(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Creator entanglement visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing creator entanglement: {e}")
        
        # Fallback visualization using soul visualizer
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                soul_viz.visualize_core_structure(
                    show=self.show_images,
                    save=self.save_to_disk
                )
                logger.info("Creator entanglement core visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing creator entanglement core structure: {e}")
    
    def visualize_harmonic_strengthening(self, soul):
        """
        Visualize a soul's harmonic strengthening.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
        
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                # Visualize frequency signature if method exists
                if hasattr(soul_viz, 'visualize_frequency_signature'):
                    soul_viz.visualize_frequency_signature(
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Frequency signature visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing frequency signature: {e}")
                
            try:
                # Visualize energy harmony if method exists
                if hasattr(soul_viz, 'visualize_energy_harmony'):
                    soul_viz.visualize_energy_harmony(
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Energy harmony visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing energy harmony: {e}")
    
    def visualize_life_cord(self, soul):
        """
        Visualize a soul's life cord.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
        
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                # Check if method exists
                if hasattr(soul_viz, 'visualize_life_cord'):
                    soul_viz.visualize_life_cord(
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Life cord visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing life cord: {e}")
    
    def visualize_earth_harmonization(self, soul):
        """
        Visualize a soul's harmonization with Earth.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
        
        # Check if journey visualizer has the dashboard method
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Try journey dashboard if available
                if hasattr(journey_viz, 'create_journey_dashboard'):
                    journey_viz.create_journey_dashboard(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Earth harmonization journey dashboard complete")
                # Fallback to timeline
                elif hasattr(journey_viz, 'visualize_journey_timeline'):
                    journey_viz.visualize_journey_timeline(
                        soul,
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Earth harmonization journey timeline complete")
            except Exception as e:
                logger.error(f"Error visualizing earth harmonization journey: {e}")
        
        # Also use soul visualizer for energy harmony
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                soul_viz.visualize_energy_harmony(
                    show=self.show_images,
                    save=self.save_to_disk
                )
                logger.info("Earth harmonization energy visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing earth harmonization energy: {e}")
    
    def visualize_identity_crystallization(self, soul):
        """
        Visualize a soul's identity crystallization.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
        
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Try journey timeline
                if hasattr(journey_viz, 'visualize_journey_timeline'):
                    journey_viz.visualize_journey_timeline(
                        soul,
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Identity crystallization journey visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing identity crystallization journey: {e}")
        
        # Also use soul visualizer for identity details
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                # Check if method exists
                if hasattr(soul_viz, 'visualize_identity'):
                    soul_viz.visualize_identity(
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Identity details visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing identity details: {e}")
    
    def visualize_pre_birth(self, soul, mother_profile_data=None):
        """
        Visualize a soul's state before birth.
        
        Args:
            soul: The SoulSpark instance
            mother_profile_data: Mother resonance data if available
        """
        if not self.enabled:
            return
            
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Try journey dashboard if available
                if hasattr(journey_viz, 'create_journey_dashboard'):
                    journey_viz.create_journey_dashboard(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Pre-birth journey dashboard complete")
                # Fallback to timeline
                elif hasattr(journey_viz, 'visualize_journey_timeline'):
                    journey_viz.visualize_journey_timeline(
                        soul,
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Pre-birth journey timeline complete")
            except Exception as e:
                logger.error(f"Error visualizing pre-birth journey: {e}")
            
            # If mother profile data exists, try to visualize mother resonance
            if mother_profile_data:
                try:
                    # Check if the method exists
                    if hasattr(journey_viz, 'visualize_mother_resonance'):
                        journey_viz.visualize_mother_resonance(
                            soul,
                            mother_profile_data=mother_profile_data,
                            show=self.show_images,
                            save=self.save_to_disk
                        )
                        logger.info("Mother resonance visualization complete")
                except Exception as e:
                    logger.error(f"Error visualizing mother resonance: {e}")
        
        # Create a soul dashboard as well
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                # Check if dashboard method exists
                if hasattr(soul_viz, 'create_soul_dashboard'):
                    soul_viz.create_soul_dashboard(
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Pre-birth soul dashboard complete")
            except Exception as e:
                logger.error(f"Error creating pre-birth soul dashboard: {e}")
    
    def visualize_birth_process(self, soul):
        """
        Visualize a soul's birth process.
        
        Args:
            soul: The SoulSpark instance
        """
        if not self.enabled:
            return
            
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Check if birth process visualization method exists
                if hasattr(journey_viz, 'visualize_birth_process'):
                    journey_viz.visualize_birth_process(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Birth process visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing birth process: {e}")
            
            try:
                # Try journey dashboard if available
                if hasattr(journey_viz, 'create_journey_dashboard'):
                    journey_viz.create_journey_dashboard(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Birth journey dashboard complete")
                # Fallback to timeline
                elif hasattr(journey_viz, 'visualize_journey_timeline'):
                    journey_viz.visualize_journey_timeline(
                        soul,
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Birth journey timeline complete")
            except Exception as e:
                logger.error(f"Error visualizing birth journey: {e}")
        
        # Create a final soul dashboard
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                # Check if dashboard method exists
                if hasattr(soul_viz, 'create_soul_dashboard'):
                    soul_viz.create_soul_dashboard(
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Final soul dashboard complete")
            except Exception as e:
                logger.error(f"Error creating final soul dashboard: {e}")
    
    def visualize_failure(self, soul, stage_name):
        """
        Visualize a soul's state after a failure.
        
        Args:
            soul: The SoulSpark instance
            stage_name: Name of the stage that failed
        """
        if not self.enabled:
            return
            
        journey_viz = self._get_journey_visualizer()
        if journey_viz:
            try:
                # Try journey dashboard if available
                if hasattr(journey_viz, 'create_journey_dashboard'):
                    journey_viz.create_journey_dashboard(
                        soul, 
                        show=self.show_images, 
                        save=self.save_to_disk
                    )
                    logger.info("Failure state journey dashboard complete")
                # Fallback to timeline
                elif hasattr(journey_viz, 'visualize_journey_timeline'):
                    journey_viz.visualize_journey_timeline(
                        soul,
                        show=self.show_images,
                        save=self.save_to_disk
                    )
                    logger.info("Failure state journey timeline complete")
            except Exception as e:
                logger.error(f"Error visualizing failure state journey: {e}")
        
        # Create soul core visualization
        soul_viz = self._get_soul_visualizer(soul)
        if soul_viz:
            try:
                soul_viz.visualize_core_structure(
                    show=self.show_images,
                    save=self.save_to_disk
                )
                logger.info("Failure state core visualization complete")
            except Exception as e:
                logger.error(f"Error visualizing failure state core: {e}")


# Create a global instance for easy import and use
visualizer = SimpleVisualizer()

def initialize_visualization(enabled=True, save_to_disk=True, show_images=False,
                            output_dir="output/visualizations"):
    """
    Initialize the visualization system with the specified settings.
    
    Args:
        enabled: Whether visualization is enabled
        save_to_disk: Whether to save visualizations to disk
        show_images: Whether to display visualizations (interactive mode)
        output_dir: Directory to save visualizations
        
    Returns:
        SimpleVisualizer: The initialized visualizer
    """
    global visualizer
    visualizer = SimpleVisualizer(
        enabled=enabled,
        save_to_disk=save_to_disk,
        show_images=show_images,
        output_dir=output_dir
    )
    return visualizer

# Simple helper functions for direct use in root_controller.py
def visualize_field_state(field_controller, stage="general"):
    visualizer.visualize_field_state(field_controller, stage)

def visualize_guff_strengthening(soul, field_controller):
    visualizer.visualize_guff_strengthening(soul, field_controller)

def visualize_sephiroth_journey(soul):
    visualizer.visualize_sephiroth_journey(soul)

def visualize_creator_entanglement(soul):
    visualizer.visualize_creator_entanglement(soul)

def visualize_harmonic_strengthening(soul):
    visualizer.visualize_harmonic_strengthening(soul)

def visualize_life_cord(soul):
    visualizer.visualize_life_cord(soul)

def visualize_earth_harmonization(soul):
    visualizer.visualize_earth_harmonization(soul)

def visualize_identity_crystallization(soul):
    visualizer.visualize_identity_crystallization(soul)

def visualize_pre_birth(soul, mother_profile_data=None):
    visualizer.visualize_pre_birth(soul, mother_profile_data)

def visualize_birth_process(soul):
    visualizer.visualize_birth_process(soul)

def visualize_failure(soul, stage_name):
    visualizer.visualize_failure(soul, stage_name)