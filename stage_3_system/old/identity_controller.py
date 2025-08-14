# identity_processing_controller.py V1
"""
Identity Processing Controller - Stage 3 System
Simple controller that runs the 3 specialized sensory data collection files.
No additional functionality - just orchestrates the collection processes.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import uuid

# Import the 3 sensory data collection modules
from stage_3_system.identity_processing.audio_identity_collection import AudioIdentityCollection
from stage_3_system.identity_processing.text_identity_collection import TextIdentityCollection
from stage_3_system.identity_processing.physical_identity_collection import PhysicalIdentityCollection

# --- Logging Setup ---
logger = logging.getLogger("IdentityProcessingController")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class IdentityProcessingController:
    """
    Simple controller for running identity processing sensory data collection.
    Orchestrates the 3 collection modules: Audio, Text, Physical.
    """
    
    def __init__(self, identity_context: Dict[str, Any] = None):
        """Initialize identity processing controller."""
        self.controller_id = str(uuid.uuid4())
        self.creation_time = datetime.now().isoformat()
        self.identity_context = identity_context or {}
        
        # Initialize collection modules
        self.audio_collection = AudioIdentityCollection(identity_context)
        self.text_collection = TextIdentityCollection(identity_context)
        self.physical_collection = PhysicalIdentityCollection(identity_context)

        # Ensure all collection module loggers propagate (they should by default)
        logging.getLogger("AudioIdentityCollection").propagate = True
        logging.getLogger("TextIdentityCollection").propagate = True
        logging.getLogger("PhysicalIdentityCollection").propagate = True

        
        
        logger.info(f"🎯 Identity processing controller initialized: {self.controller_id[:8]}")
    
    def run_identity_processing(self, process_name: str = "identity_crystallization") -> Dict[str, Any]:
        """Run all 3 identity processing collections."""
        logger.info(f"🚀 Starting identity processing: {process_name}")
        
        processing_session = {
            'session_id': str(uuid.uuid4()),
            'controller_id': self.controller_id,
            'process_name': process_name,
            'start_time': datetime.now().isoformat(),
            'collections_attempted': 0,
            'collections_successful': 0,
            'collection_results': {}
        }
        
        try:
            # Run Audio Collection
            logger.info("1️⃣ Running audio identity collection...")
            processing_session['collections_attempted'] += 1
            try:
                audio_result = self.audio_collection.run_audio_identity_capture(process_name)
                processing_session['collection_results']['audio'] = audio_result
                if audio_result.get('success', False):
                    processing_session['collections_successful'] += 1
                logger.info(f"   ✅ Audio collection: {audio_result.get('success', False)}")
            except Exception as e:
                processing_session['collection_results']['audio'] = {'success': False, 'error': str(e)}
                logger.warning(f"   ⚠️ Audio collection failed: {e}")
            
            # Run Text Collection
            logger.info("2️⃣ Running text identity collection...")
            processing_session['collections_attempted'] += 1
            try:
                text_result = self.text_collection.run_text_identity_capture(process_name)
                processing_session['collection_results']['text'] = text_result
                if text_result.get('success', False):
                    processing_session['collections_successful'] += 1
                logger.info(f"   ✅ Text collection: {text_result.get('success', False)}")
            except Exception as e:
                processing_session['collection_results']['text'] = {'success': False, 'error': str(e)}
                logger.warning(f"   ⚠️ Text collection failed: {e}")
            
            # Run Physical Collection  
            logger.info("3️⃣ Running physical identity collection...")
            processing_session['collections_attempted'] += 1
            try:
                physical_result = self.physical_collection.run_physical_identity_capture(process_name)
                processing_session['collection_results']['physical'] = physical_result
                if physical_result.get('success', False):
                    processing_session['collections_successful'] += 1
                logger.info(f"   ✅ Physical collection: {physical_result.get('success', False)}")
            except Exception as e:
                processing_session['collection_results']['physical'] = {'success': False, 'error': str(e)}
                logger.warning(f"   ⚠️ Physical collection failed: {e}")
            
            # Finalize session
            processing_session.update({
                'end_time': datetime.now().isoformat(),
                'success_rate': processing_session['collections_successful'] / processing_session['collections_attempted'],
                'overall_success': processing_session['collections_successful'] >= 3  # All 3 required
            })
            
            logger.info(f"🏁 Identity processing completed")
            logger.info(f"   Success rate: {processing_session['success_rate']:.1%}")
            logger.info(f"   Collections successful: {processing_session['collections_successful']}/3")
            
            return processing_session
            
        except Exception as e:
            processing_session.update({
                'end_time': datetime.now().isoformat(),
                'overall_success': False,
                'error': str(e)
            })
            logger.error(f"❌ Identity processing failed: {e}")
            raise RuntimeError(f"Identity processing failed: {e}") from e
    
    def run_audio_only(self, process_name: str = "audio_test") -> Dict[str, Any]:
        """Run only audio collection for testing."""
        logger.info(f"🎵 Running audio-only collection: {process_name}")
        try:
            return self.audio_collection.run_audio_identity_capture(process_name)
        except Exception as e:
            logger.error(f"Audio-only collection failed: {e}")
            raise RuntimeError(f"Audio-only collection failed: {e}") from e
    
    def run_text_only(self, process_name: str = "text_test") -> Dict[str, Any]:
        """Run only text collection for testing."""
        logger.info(f"📝 Running text-only collection: {process_name}")
        try:
            return self.text_collection.run_text_identity_capture(process_name)
        except Exception as e:
            logger.error(f"Text-only collection failed: {e}")
            raise RuntimeError(f"Text-only collection failed: {e}") from e
    
    def run_physical_only(self, process_name: str = "physical_test") -> Dict[str, Any]:
        """Run only physical collection for testing."""
        logger.info(f"💻 Running physical-only collection: {process_name}")
        try:
            return self.physical_collection.run_physical_identity_capture(process_name)
        except Exception as e:
            logger.error(f"Physical-only collection failed: {e}")
            raise RuntimeError(f"Physical-only collection failed: {e}") from e
    
    def get_controller_status(self) -> Dict[str, Any]:
        """Get controller status and collection module states."""
        return {
            'controller_id': self.controller_id,
            'creation_time': self.creation_time,
            'identity_context': self.identity_context,
            'modules_initialized': {
                'audio_collection': self.audio_collection is not None,
                'text_collection': self.text_collection is not None,
                'physical_collection': self.physical_collection is not None
            }
        }


