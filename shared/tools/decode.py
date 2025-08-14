# shared/tools/decode.py  
"""
Image Decoding Tools - EXIF and Steganography
Handles decoding data from images using EXIF metadata and LSB steganography
"""

import json
import piexif
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger('ImageDecoder')

class ImageDecoder:
    """Decodes data from images using multiple methods"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff']
    
    def decode_exif_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Decode metadata from image EXIF data
        
        Args:
            image_path: Path to encoded image
            
        Returns:
            Decoded metadata or None if not found
        """
        try:
            # Open image and get EXIF
            img = Image.open(image_path)
            exif_bytes = img.info.get("exif")
            
            if not exif_bytes:
                return None
            
            exif_data = piexif.load(exif_bytes)
            
            # Check software tag
            software = exif_data["0th"].get(305)
            if software != b'SoulGlyphSystem':
                return None
            
            # Get user comment
            user_comment = exif_data["Exif"].get(37510)
            if user_comment is None:
                return None
            
            # Decode JSON
            if isinstance(user_comment, bytes):
                json_str = user_comment.decode("utf-8")
            else:
                json_str = user_comment
            
            metadata = json.loads(json_str)
            
            logger.info(f"Successfully decoded EXIF metadata from {image_path}")
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to decode EXIF metadata: {e}")
            return None
    
    def decode_lsb_steganography(self, image_path: str) -> Optional[str]:
        """
        Decode data using LSB steganography
        
        Args:
            image_path: Path to encoded image
            
        Returns:
            Decoded string data or None if not found
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            img_flat = img_array.flatten()
            
            # Extract length (first 32 bits)
            length_bits = ''.join([str(b & 1) for b in img_flat[:32]])
            data_length = int(length_bits, 2)
            
            # Validate length
            if data_length <= 0 or data_length > len(img_flat) - 32:
                return None
            
            # Extract data bits
            data_bits = ''.join([str(b & 1) for b in img_flat[32:32+data_length]])
            
            # Convert binary to string
            hidden_data = ''
            for i in range(0, len(data_bits), 8):
                byte = data_bits[i:i+8]
                if len(byte) == 8:
                    hidden_data += chr(int(byte, 2))
            
            logger.info(f"Successfully decoded LSB steganography from {image_path}")
            return hidden_data
            
        except Exception as e:
            logger.warning(f"Failed to decode LSB steganography: {e}")
            return None
    
    def decode_full_data(self, image_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Decode both EXIF metadata and LSB steganography
        
        Args:
            image_path: Path to encoded image
            
        Returns:
            Tuple of (metadata, hidden_data) - either can be None if not found
        """
        metadata = self.decode_exif_metadata(image_path)
        hidden_data = self.decode_lsb_steganography(image_path)
        
        logger.info(f"Decoded full data from {image_path}: "
                   f"EXIF={'found' if metadata else 'not found'}, "
                   f"LSB={'found' if hidden_data else 'not found'}")
        
        return metadata, hidden_data
    
    def verify_soul_glyph(self, image_path: str) -> bool:
        """
        Verify if image is a Soul Glyph System encoded image
        
        Args:
            image_path: Path to image
            
        Returns:
            True if verified Soul Glyph image
        """
        try:
            img = Image.open(image_path)
            exif_bytes = img.info.get("exif")
            
            if not exif_bytes:
                return False
            
            exif_data = piexif.load(exif_bytes)
            software = exif_data["0th"].get(305)
            
            return software == b'SoulGlyphSystem'
            
        except Exception:
            return False
        
def create_image_decoder() -> ImageDecoder:
    """Create a new image decoder instance"""
    return ImageDecoder()