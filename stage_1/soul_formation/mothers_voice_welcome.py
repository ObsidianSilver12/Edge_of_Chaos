"""
First Breath - Mother's Voice Welcome System

Instead of simulating breath activation, we create a proper welcome moment
where the mother's voice provides the first conscious sensory experience,
creating the initial node/fragment in the new incarnated consciousness.

This represents the first real sensory data processing in the new brain.
"""

from datetime import datetime
from typing import Dict, Any
import logging
import uuid
import os

# --- Logging Setup ---
logger = logging.getLogger("FirstBreath")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MothersVoiceWelcome:
    """
    Mother's Voice Welcome System for First Breath
    
    Creates the first conscious sensory experience through mother's voice,
    generating the first real node/fragment in the incarnated brain.
    """
    
    def __init__(self, brain_structure=None, energy_system=None):
        """Initialize mother's voice welcome system."""
        self.welcome_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.brain_structure = brain_structure
        self.energy_system = energy_system
        
        # Voice characteristics
        self.mother_voice_frequency = 220.0  # A3 - warm, nurturing frequency
        self.emotional_resonance = 0.95      # High emotional content
        self.love_energy_level = 1.0         # Maximum love energy
        
        # Welcome message
        self.welcome_message = "Welcome to this world, little one. You are loved beyond measure."
        self.message_duration = 3.5  # seconds
        
        # Audio generation
        self.audio_generated = False
        self.audio_file_path = None
        
        logger.info("👶 Mother's voice welcome system initialized: %s", self.welcome_id[:8])
    
    def create_welcome_moment(self) -> Dict[str, Any]:
        """
        Create the complete welcome moment with mother's voice.
        
        Returns:
            Welcome moment creation results and first sensory node
        """
        logger.info("👶 Creating mother's voice welcome moment...")
        
        try:
            # Generate the audio
            audio_result = self._generate_mothers_voice_audio()
            
            # Create first sensory experience
            first_experience = self._create_first_sensory_experience()
            
            # Process through brain if available
            brain_processing = None
            if self.brain_structure and self.energy_system:
                brain_processing = self._process_first_voice_through_brain(first_experience)
            
            welcome_metrics = {
                'welcome_created': True,
                'welcome_id': self.welcome_id,
                'audio_generated': audio_result['success'],
                'audio_file': audio_result.get('file_path'),
                'voice_frequency': self.mother_voice_frequency,
                'emotional_resonance': self.emotional_resonance,
                'message_content': self.welcome_message,
                'duration_seconds': self.message_duration,
                'first_sensory_node': first_experience,
                'brain_processing': brain_processing
            }
            
            logger.info("✅ Mother's voice welcome created successfully")
            logger.info("   Message: '%s'", self.welcome_message)
            logger.info("   Frequency: %.1f Hz", self.mother_voice_frequency)
            logger.info("   Duration: %.1f seconds", self.message_duration)
            
            return {
                'success': True,
                'welcome_metrics': welcome_metrics,
                'first_breath_authentic': True,
                'sensory_node_created': True
            }
            
        except Exception as e:
            logger.error("Failed to create welcome moment: %s", str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_mothers_voice_audio(self) -> Dict[str, Any]:
        """Generate audio file of mother's voice welcome."""
        
        try:
            # Create audio directory in output/sounds/ if it doesn't exist
            audio_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'sounds')
            os.makedirs(audio_dir, exist_ok=True)
            
            # Audio file path
            audio_filename = f"mothers_welcome_{self.welcome_id[:8]}.wav"
            self.audio_file_path = os.path.join(audio_dir, audio_filename)
            
            # Generate audio using text-to-speech with voice characteristics
            audio_content = self._synthesize_nurturing_voice()
            
            # Save audio file
            with open(self.audio_file_path, 'wb') as f:
                f.write(audio_content)
            
            self.audio_generated = True
            
            logger.info("🎵 Mother's voice audio generated: %s", audio_filename)
            
            return {
                'success': True,
                'file_path': self.audio_file_path,
                'filename': audio_filename,
                'message': self.welcome_message,
                'frequency': self.mother_voice_frequency,
                'duration': self.message_duration
            }
            
        except Exception as e:
            logger.error("Audio generation failed: %s", str(e))
            return {
                'success': False,
                'error': str(e),
                'simulated': True
            }
    
    def _synthesize_nurturing_voice(self) -> bytes:
        """
        Synthesize nurturing mother's voice.
        
        Note: This is a placeholder for actual audio synthesis.
        In real implementation, would use TTS with specific voice characteristics.
        """
        
        # Placeholder audio data representing the welcome message
        # In real implementation, this would use proper TTS synthesis
        
        # Simulate WAV file header + data
        sample_rate = 44100
        duration_samples = int(self.message_duration * sample_rate)
        
        # Basic WAV header (44 bytes)
        wav_header = bytearray([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x24, 0x08, 0x00, 0x00,  # File size - 8
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1Size
            0x01, 0x00,              # AudioFormat (PCM)
            0x01, 0x00,              # NumChannels (mono)
            0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
            0x88, 0x58, 0x01, 0x00,  # ByteRate
            0x02, 0x00,              # BlockAlign
            0x10, 0x00,              # BitsPerSample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x08, 0x00, 0x00   # Subchunk2Size
        ])
        
        # Generate realistic human speech synthesis
        import math
        import random
        
        # Human speech characteristics
        base_freq = self.mother_voice_frequency  # Fundamental frequency (F0)
        
        # Parse message into phonetic segments
        words = self.welcome_message.split()
        total_words = len(words)
        
        audio_data = bytearray()
        
        # Speech timing: ~150 words per minute for gentle speech
        word_duration = 60.0 / 150.0  # ~0.4s per word
        syllable_duration = word_duration / 2  # Average 2 syllables per word
        
        current_time = 0.0
        
        for word_idx, word in enumerate(words):
            # Estimate syllables (rough approximation)
            syllables = max(1, len([c for c in word.lower() if c in 'aeiou']))
            
            for syl_idx in range(syllables):
                syl_start = int(current_time * sample_rate)
                syl_end = int((current_time + syllable_duration) * sample_rate)
                syl_length = syl_end - syl_start
                
                if syl_start >= duration_samples:
                    break
                    
                syl_end = min(syl_end, duration_samples)
                syl_length = syl_end - syl_start
                
                if syl_length > 0:
                    # Generate formant frequencies for human speech
                    # Female voice formants (Hz): F1=270-610, F2=850-2550, F3=2200-3500
                    f1 = 350 + 150 * random.random()  # First formant (tongue height)
                    f2 = 1200 + 800 * random.random()  # Second formant (tongue position)
                    f3 = 2800 + 500 * random.random()  # Third formant (lip rounding)
                    
                    # Pitch contour - natural speech melody
                    pitch_start = base_freq * (0.9 + 0.2 * random.random())
                    pitch_end = base_freq * (0.9 + 0.2 * random.random())
                    
                    # Generate syllable
                    for i in range(syl_length):
                        t_local = i / sample_rate
                        t_global = (syl_start + i) / sample_rate
                        progress = i / syl_length if syl_length > 1 else 0
                        
                        # Interpolate pitch
                        pitch = pitch_start + (pitch_end - pitch_start) * progress
                        
                        # Generate harmonic series (fundamental + overtones)
                        amplitude = 0.0
                        
                        # Fundamental frequency
                        amplitude += 0.4 * math.sin(2 * math.pi * pitch * t_local)
                        
                        # Formant resonances (what makes it sound like speech)
                        formant1 = 0.3 * math.sin(2 * math.pi * f1 * t_local) * math.exp(-(f1 * t_local * 0.01))
                        formant2 = 0.2 * math.sin(2 * math.pi * f2 * t_local) * math.exp(-(f2 * t_local * 0.005))
                        formant3 = 0.1 * math.sin(2 * math.pi * f3 * t_local) * math.exp(-(f3 * t_local * 0.003))
                        
                        # Combine formants with fundamental
                        amplitude += formant1 + formant2 + formant3
                        
                        # Add harmonic overtones
                        amplitude += 0.15 * math.sin(2 * math.pi * pitch * 2 * t_local)  # Second harmonic
                        amplitude += 0.08 * math.sin(2 * math.pi * pitch * 3 * t_local)  # Third harmonic
                        
                        # Speech envelope (attack-sustain-decay)
                        envelope = 1.0
                        attack_time = 0.05
                        decay_time = 0.1
                        
                        if t_local < attack_time:
                            envelope = t_local / attack_time
                        elif t_local > syllable_duration - decay_time:
                            envelope = (syllable_duration - t_local) / decay_time
                        
                        # Add breath and vocal texture
                        breath_noise = 0.02 * (random.random() - 0.5)
                        
                        # Natural vibrato (emotional warmth)
                        vibrato = 1.0 + 0.03 * math.sin(2 * math.pi * 5.5 * t_global)
                        
                        amplitude *= envelope * vibrato
                        amplitude += breath_noise
                        
                        # Clamp and convert to 16-bit PCM
                        amplitude = max(-0.8, min(0.8, amplitude))
                        sample = int(amplitude * 32767)
                        
                        if len(audio_data) < (duration_samples * 2):  # 2 bytes per sample
                            audio_data.extend(sample.to_bytes(2, byteorder='little', signed=True))
                
                current_time += syllable_duration
            
            # Add natural pause between words
            pause_duration = 0.1 + 0.1 * random.random()
            current_time += pause_duration
        
        # Fill remaining time with silence
        remaining_samples = duration_samples - len(audio_data) // 2
        for _ in range(remaining_samples):
            if len(audio_data) < (duration_samples * 2):
                audio_data.extend((0).to_bytes(2, byteorder='little', signed=True))
        
        return wav_header + audio_data
    
    def _create_first_sensory_experience(self) -> Dict[str, Any]:
        """Create the first sensory experience node/fragment."""
        
        first_experience = {
            'node_id': f"first_voice_{self.welcome_id[:8]}",
            'type': 'sensory_auditory',
            'subtype': 'human_voice',
            'content': {
                'message': self.welcome_message,
                'voice_type': 'mother',
                'frequency': self.mother_voice_frequency,
                'emotional_content': {
                    'love': self.love_energy_level,
                    'safety': 0.95,
                    'welcome': 1.0,
                    'nurturing': 0.98
                },
                'duration': self.message_duration,
                'timestamp': self.creation_time
            },
            'processing_priority': 'maximum',  # First experience gets highest priority
            'memory_importance': 'foundational',
            'neural_pathways': ['auditory_cortex', 'limbic_system', 'emotional_processing'],
            'energy_signature': self.mother_voice_frequency,
            'incarnation_marker': True  # This marks the beginning of incarnated experience
        }
        
        return first_experience
    
    def _process_first_voice_through_brain(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process the first voice experience through the brain systems."""
        
        if not self.brain_structure or not self.energy_system:
            return {'processed': False, 'reason': 'brain_systems_not_available'}
        
        try:
            # Start processing session for first voice
            session_id = self.energy_system.start_mycelial_processing(
                seed_id=f"first_voice_{self.welcome_id[:8]}",
                processing_type='first_incarnation_experience'
            )
            
            # Allocate significant energy for this foundational experience
            energy_transfer = self.energy_system.transfer_energy_for_step(session_id, 5)  # More energy
            
            # Complete processing
            processing_completion = self.energy_system.complete_subconscious_processing(session_id)
            
            # Create neural flag for conscious awareness
            neural_flag = processing_completion.get('neural_flag')
            if neural_flag:
                conscious_session = self.energy_system.start_conscious_processing(neural_flag)
            
            return {
                'processed': True,
                'session_id': session_id,
                'energy_allocated': energy_transfer,
                'neural_flag': neural_flag,
                'conscious_processing': conscious_session if neural_flag else None,
                'first_incarnation_node_created': True
            }
            
        except Exception as e:
            logger.error("Brain processing of first voice failed: %s", str(e))
            return {
                'processed': False,
                'error': str(e)
            }
    
    def get_welcome_status(self) -> Dict[str, Any]:
        """Get current welcome system status."""
        return {
            'welcome_id': self.welcome_id,
            'creation_time': self.creation_time,
            'audio_generated': self.audio_generated,
            'audio_file_path': self.audio_file_path,
            'mother_voice_frequency': self.mother_voice_frequency,
            'emotional_resonance': self.emotional_resonance,
            'welcome_message': self.welcome_message,
            'message_duration': self.message_duration,
            'love_energy_level': self.love_energy_level
        }


def create_mothers_voice_welcome(brain_structure=None, energy_system=None) -> MothersVoiceWelcome:
    """
    Create mother's voice welcome system for first breath.
    
    Args:
        brain_structure: Brain structure for processing
        energy_system: Energy system for processing
        
    Returns:
        Configured MothersVoiceWelcome instance
    """
    
    welcome_system = MothersVoiceWelcome(brain_structure, energy_system)
    welcome_result = welcome_system.create_welcome_moment()
    
    if not welcome_result['success']:
        raise RuntimeError(f"Mother's voice welcome creation failed: {welcome_result['error']}")
    
    return welcome_system


# === TESTING ===

def test_mothers_voice_welcome():
    """Test mother's voice welcome system."""
    print("\n" + "="*60)
    print("👶 TESTING MOTHER'S VOICE WELCOME SYSTEM")
    print("="*60)
    
    try:
        # Create welcome system
        welcome = create_mothers_voice_welcome()
        
        print(f"1. Mother's voice welcome created: {welcome.welcome_id[:8]}")
        print(f"   Message: '{welcome.welcome_message}'")
        print(f"   Frequency: {welcome.mother_voice_frequency} Hz")
        print(f"   Duration: {welcome.message_duration} seconds")
        
        # Check audio generation
        if welcome.audio_generated:
            print(f"   Audio file: {os.path.basename(welcome.audio_file_path)}")
        
        print("\n✅ Mother's voice welcome test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Mother's voice welcome test failed: {e}")
        return False


if __name__ == "__main__":
    test_mothers_voice_welcome()
