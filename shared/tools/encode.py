# shared/tools/encode.py
"""
Image Encoding Tools - EXIF and Steganography
Handles encoding data into images using EXIF metadata and LSB steganography
"""

import json
import piexif
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger('ImageEncoder')

class ImageEncoder:
    """Encodes data into images using multiple methods"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff']
    
    def encode_exif_metadata(self, image_path: str, metadata: Dict[str, Any], output_path: str) -> str:
        """
        Encode metadata into image EXIF data
        
        Args:
            image_path: Path to source image
            metadata: Dictionary of metadata to encode
            output_path: Path for encoded image
            
        Returns:
            Path to encoded image
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Create EXIF dictionary
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            # Set software tag to identify our system
            exif_dict["0th"][305] = b'SoulGlyphSystem'
            
            # Encode metadata as JSON in UserComment field
            json_data = json.dumps(metadata, ensure_ascii=False)
            user_comment_bytes = json_data.encode('utf-8')
            exif_dict["Exif"][37510] = user_comment_bytes
            
            # Add creation timestamp
            from datetime import datetime
            exif_dict["0th"][306] = datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode('ascii')
            
            # Dump EXIF to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Save image with EXIF
            img.save(output_path, exif=exif_bytes)
            
            logger.info(f"Encoded EXIF metadata into {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to encode EXIF metadata: {e}")
            raise
    
    def encode_lsb_steganography(self, image_path: str, hidden_data: str, output_path: str) -> str:
        """
        Encode data using LSB steganography
        
        Args:
            image_path: Path to source image
            hidden_data: String data to hide
            output_path: Path for encoded image
            
        Returns:
            Path to encoded image
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convert data to binary
            binary_data = ''.join(format(ord(char), '08b') for char in hidden_data)
            data_length = len(binary_data)
            
            # Add length header (32 bits)
            length_binary = format(data_length, '032b')
            full_binary = length_binary + binary_data
            
            # Flatten image array
            img_flat = img_array.flatten()
            
            # Check if image is large enough
            if len(full_binary) > len(img_flat):
                raise ValueError("Image too small for data")
            
            # Modify LSBs
            for i, bit in enumerate(full_binary):
                img_flat[i] = (img_flat[i] & 0xFE) | int(bit)
            
            # Reshape and save
            encoded_array = img_flat.reshape(img_array.shape)
            encoded_img = Image.fromarray(encoded_array.astype(np.uint8))
            encoded_img.save(output_path)
            
            logger.info(f"Encoded LSB steganography into {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to encode LSB steganography: {e}")
            raise
    
    def encode_full_data(self, image_path: str, metadata: Dict[str, Any], 
                        hidden_data: str, output_path: str) -> str:
        """
        Encode both EXIF metadata and LSB steganography
        
        Args:
            image_path: Path to source image
            metadata: Dictionary of metadata for EXIF
            hidden_data: String data for steganography
            output_path: Path for final encoded image
            
        Returns:
            Path to encoded image
        """
        try:
            # First encode steganography
            temp_path = output_path.replace('.', '_temp.')
            self.encode_lsb_steganography(image_path, hidden_data, temp_path)
            
            # Then add EXIF metadata
            final_path = self.encode_exif_metadata(temp_path, metadata, output_path)
            
            # Clean up temp file
            import os
            os.remove(temp_path)
            
            logger.info(f"Encoded full data (EXIF + LSB) into {output_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to encode full data: {e}")
            raise


# Factory functions
def create_image_encoder() -> ImageEncoder:
    """Create a new image encoder instance"""
    return ImageEncoder()

