"""
Sephiroth Sound Integration Module

This module integrates the Sephiroth sounds with the field system.
It provides functions to enhance field formation, resonance, and stability
through specialized Sephiroth-specific sound patterns.

Author: Soul Development Framework Team
"""

import logging
import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Sephiroth sound generator
try:
    from sounds.sounds_of_sephiroth import SephirothSoundGenerator, extend_field_system_with_sephiroth_sounds
except ImportError:
    logging.warning("sounds_of_sephiroth.py not available. Sephiroth sound integration will not function.")
    SephirothSoundGenerator = None
    extend_field_system_with_sephiroth_sounds = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sephiroth_sound_integration.log'
)
logger = logging.getLogger('sephiroth_sound_integration')

class SephirothSoundIntegration:
    """
    Integrates Sephiroth sounds with field systems to enhance formation and stability.
    """
    
    def __init__(self, base_frequency=432.0, output_dir="output/sounds/sephiroth"):
        """
        Initialize the Sephiroth sound integration system.
        
        Args:
            base_frequency (float): Base frequency for sound generation
            output_dir (str): Directory for saving generated sounds
        """
        # Create Sephiroth sound generator
        if SephirothSoundGenerator is not None:
            self.sound_generator = SephirothSoundGenerator(base_frequency, output_dir)
            logger.info("Sephiroth sound generator initialized")
        else:
            self.sound_generator = None
            logger.error("SephirothSoundGenerator not available. Integration will not function.")
        
        # Track integrated field systems
        self.integrated_fields = {}
    
    def integrate_with_field(self, field_system, field_name):
        """
        Integrate Sephiroth sounds with a field system.
        
        Args:
            field_system: The field system to integrate with
            field_name (str): Name identifier for the field
            
        Returns:
            bool: True if integration was successful
        """
        if self.sound_generator is None:
            logger.error("Cannot integrate: sound generator not available")
            return False
        
        if field_system is None:
            logger.error("Cannot integrate with None field system")
            return False
        
        try:
            # Attempt to extend the field system
            if extend_field_system_with_sephiroth_sounds is not None:
                success = extend_field_system_with_sephiroth_sounds(field_system)
                
                if success:
                    # Track this integration
                    self.integrated_fields[field_name] = field_system
                    logger.info(f"Successfully integrated Sephiroth sounds with {field_name}")
                    return True
                else:
                    logger.warning(f"Failed to extend field system {field_name}")
                    return False
            else:
                logger.error("extend_field_system_with_sephiroth_sounds function not available")
                return False
                
        except Exception as e:
            logger.error(f"Error integrating with field {field_name}: {str(e)}")
            return False
    
    def generate_sephiroth_field_sound(self, sephirah, duration=30.0, save=True):
        """
        Generate a sound for a Sephiroth field directly.
        
        Args:
            sephirah (str): Name of the Sephiroth
            duration (float): Duration in seconds
            save (bool): Whether to save the sound to file
            
        Returns:
            numpy.ndarray or str: Sound waveform or file path if saved
        """
        if self.sound_generator is None:
            logger.error("Cannot generate field sound: sound generator not available")
            return None
        
        try:
            # Generate the field activation sound
            sound = self.sound_generator.generate_sephiroth_field_activation(sephirah, duration)
            
            if sound is None:
                return None
                
            if save:
                # Save to file
                filepath = self.sound_generator.save_sephiroth_field_activation(sephirah, duration)
                return filepath
            else:
                return sound
                
        except Exception as e:
            logger.error(f"Error generating Sephiroth field sound: {str(e)}")
            return None
    
    def enhance_field_resonance(self, field_system, sephirah):
        """
        Enhance a field's resonance using Sephiroth sounds.
        
        Args:
            field_system: The field system to enhance
            sephirah (str): Name of the Sephiroth to align with
            
        Returns:
            bool: True if enhancement was successful
        """
        if self.sound_generator is None or field_system is None:
            return False
        
        try:
            # Check if the field has been integrated
            field_name = getattr(field_system, 'field_name', 'unknown_field')
            
            if field_name not in self.integrated_fields:
                # Try to integrate first
                success = self.integrate_with_field(field_system, field_name)
                if not success:
                    logger.warning(f"Could not integrate with {field_name} before enhancing")
                    return False
            
            # Generate Sephiroth field sound (without saving)
            sound = self.sound_generator.generate_sephiroth_field_activation(sephirah, duration=20.0)
            
            if sound is None:
                return False
            
            # Apply sound to field if it has the method
            if hasattr(field_system, 'apply_sound_to_field'):
                # Check if the sound generator object is available
                if hasattr(self.sound_generator, 'sound_generator') and self.sound_generator.sound_generator:
                    # Create a temporary sound object for application
                    temp_sound = self.sound_generator.sound_generator.create_sound(
                        name=f"{sephirah.capitalize()} Enhancement",
                        fundamental_frequency=self.sound_generator.frequencies.get_frequency(sephirah)
                    )
                    
                    field_system.apply_sound_to_field(temp_sound, intensity=0.7)
                    logger.info(f"Enhanced {field_name} with {sephirah} resonance")
                    return True
                else:
                    logger.warning("Sound generator object not available for field application")
                    return False
            else:
                logger.warning(f"Field {field_name} does not support sound application")
                return False
                
        except Exception as e:
            logger.error(f"Error enhancing field resonance: {str(e)}")
            return False
    
    def create_sephiroth_field(self, field_system_class, sephirah, dimensions=(64, 64, 64)):
        """
        Create a new field specifically tuned to a Sephiroth.
        
        Args:
            field_system_class: The class to use for creating the field
            sephirah (str): Name of the Sephiroth to create a field for
            dimensions (tuple): Dimensions for the new field
            
        Returns:
            The created field system or None if creation failed
        """
        if self.sound_generator is None:
            logger.error("Cannot create Sephiroth field: sound generator not available")
            return None
        
        try:
            # Get the Sephiroth frequency
            frequency = self.sound_generator.frequencies.get_frequency(sephirah)
            
            # Create field name
            field_name = f"{sephirah.capitalize()} Field"
            
            # Create the field system
            field = field_system_class(
                dimensions=dimensions,
                field_name=field_name,
                base_frequency=frequency
            )
            
            # Integrate Sephiroth sounds
            success = self.integrate_with_field(field, field_name)
            
            if success:
                logger.info(f"Created {field_name} at {frequency:.2f}Hz")
                
                # Try to initialize with Sephiroth-specific frequencies
                if hasattr(field, 'add_resonance_frequency'):
                    # Add primary frequency
                    field.add_resonance_frequency(frequency, amplitude=1.0)
                    
                    # Add phi-related frequencies
                    field.add_resonance_frequency(frequency * 1.618, amplitude=0.8, is_harmonic=True)
                    field.add_resonance_frequency(frequency / 1.618, amplitude=0.7, is_harmonic=True)
                    
                    # Add specific harmonics based on Sephiroth
                    harmonics = self.sound_generator._get_sephirah_harmonic_ratios(sephirah)
                    
                    for i, ratio in enumerate(harmonics[1:]):  # Skip the fundamental
                        field.add_resonance_frequency(
                            frequency * ratio,
                            amplitude=0.9 / (i + 2),  # Decreasing amplitude
                            is_harmonic=True
                        )
                
                # Apply resonance if available
                if hasattr(field, 'apply_resonance_to_field'):
                    field.apply_resonance_to_field()
                
                return field
            else:
                logger.warning(f"Could not integrate with created {field_name}")
                return field  # Return anyway, even without integration
                
        except Exception as e:
            logger.error(f"Error creating Sephiroth field: {str(e)}")
            return None
    
    def generate_sephiroth_connection_sound(self, source_sephirah, target_sephirah, duration=20.0, save=True):
        """
        Generate a sound that creates a connection between two Sephiroth fields.
        
        This is useful for establishing pathways between Sephiroth dimensions.
        
        Args:
            source_sephirah (str): Source Sephiroth name
            target_sephirah (str): Target Sephiroth name
            duration (float): Duration in seconds
            save (bool): Whether to save the sound to file
            
        Returns:
            numpy.ndarray or str: Sound waveform or file path if saved
        """
        if self.sound_generator is None:
            logger.error("Cannot generate connection sound: sound generator not available")
            return None
        
        try:
            # Generate path sound
            sound = self.sound_generator.generate_path_sound(source_sephirah, target_sephirah, duration)
            
            if sound is None:
                return None
                
            if save:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{source_sephirah.lower()}_to_{target_sephirah.lower()}_connection_{timestamp}.wav"
                
                # Save to file
                filepath = self.sound_generator.sound_generator.save_sound(
                    sound,
                    filename=filename,
                    description=f"Connection {source_sephirah.capitalize()} to {target_sephirah.capitalize()}"
                )
                return filepath
            else:
                return sound
                
        except Exception as e:
            logger.error(f"Error generating connection sound: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # This section would typically be used to test the integration
    # with an actual field system, but we'll just print info for now
    
    integration = SephirothSoundIntegration()
    
    if integration.sound_generator is not None:
        print("Sephiroth Sound Integration initialized successfully")
        print("\nAvailable Sephiroth frequencies:")
        
        for sephirah, freq in integration.sound_generator.frequencies.frequencies.items():
            print(f"  {sephirah.capitalize()}: {freq:.2f} Hz")
        
        print("\nIntegration ready to use with field systems")
        print("Use integrate_with_field() to add Sephiroth sounds to a field")
    else:
        print("Sephiroth Sound Integration initialization failed")
        print("Check that sounds_of_sephiroth.py is available")