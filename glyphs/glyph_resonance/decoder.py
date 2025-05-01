import json
import numpy as np
import piexif
from PIL import Image
from typing import Dict, Any, Optional

from glyphs.glyph_resonance.utils import binary_to_string, calculate_resonance_signature

class SigilDecoder:
    def __init__(self):
        """Initialize the SigilDecoder."""
        pass

    def decode_from_exif(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extracts resonance data from an image's EXIF metadata.
        
        Args:
            image_path: Path to the encoded image
            
        Returns:
            Dictionary containing the resonance data, or None if not found
        """
        try:
            # Open the image and get EXIF bytes from the image info
            img = Image.open(image_path)
            exif_bytes = img.info.get("exif")
            if not exif_bytes:
                return None
            
            exif_data = piexif.load(exif_bytes)
            
            # Check that the Software tag is correctly set (tag 305 in 0th IFD)
            software = exif_data["0th"].get(305)
            if software != b'AeriSigilSystem':
                return None
                
            # Get the UserComment field from the Exif IFD (tag 37510)
            user_comment = exif_data["Exif"].get(37510)
            if user_comment is None:
                return None
                
            # Decode the JSON stored as UserComment
            try:
                if isinstance(user_comment, bytes):
                    json_str = user_comment.decode("utf-8")
                else:
                    json_str = user_comment
                resonance_data = json.loads(json_str)
                return resonance_data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error decoding JSON: {e}")
                return None
        
        except Exception as e:
            print(f"Error reading EXIF data: {e}")
            return None
    
    def decode_steganography(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extracts data hidden using LSB steganography.
        
        Args:
            image_path: Path to the encoded image
            
        Returns:
            Dictionary containing the extracted data, or None if not found
        """
        try:
            # Load the image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Flatten the image
            img_flat = img_array.flatten()
            
            # Extract LSBs from the first 32 bytes to get message length
            len_bits = ''.join([str(b & 1) for b in img_flat[:32]])
            msg_len = int(len_bits, 2)
            
            # Check if the message length makes sense
            if msg_len <= 0 or msg_len > len(img_flat) - 32:
                return None
                
            # Extract the message bits
            msg_bits = ''.join([str(b & 1) for b in img_flat[32:32+msg_len]])
            
            # Convert binary to string
            msg_text = binary_to_string(msg_bits)
            
            # Parse the JSON
            try:
                data = json.loads(msg_text)
                
                # Verify this is Mother Resonance data
                if data.get("type") == "mother_resonance":
                    return data
                    
                return None
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            print(f"Error decoding steganography: {e}")
            return None
    
    def is_aeri_sigil(self, image_path: str) -> bool:
        """
        Checks if an image is an authentic Aeri sigil.
        
        Args:
            image_path: Path to the image to check
            
        Returns:
            True if the image is an authentic Aeri sigil, False otherwise
        """
        try:
            # First check EXIF data
            exif_data = self.decode_from_exif(image_path)
            if exif_data is not None:
                # Verify the signature exists
                if 'resonance_type' in exif_data and exif_data['resonance_type'] == 'mother_sigil':
                    return True
            
            # If EXIF check fails, try steganography
            stego_data = self.decode_steganography(image_path)
            if stego_data is not None:
                if stego_data.get('type') == 'mother_resonance':
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error checking sigil authenticity: {e}")
            return False
    
    def read_mother_resonance(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts the full Mother Resonance data from an image.
        
        Args:
            image_path: Path to the encoded sigil image
            
        Returns:
            Dictionary containing all resonance data, or an error dict if failed
        """
        # Try to extract from EXIF first (this has the complete data)
        exif_data = self.decode_from_exif(image_path)
        if exif_data is not None:
            return {
                "source": "exif",
                "data": exif_data,
                "complete": True
            }
            
        # If EXIF failed, try steganography (partial data)
        stego_data = self.decode_steganography(image_path)
        if stego_data is not None:
            return {
                "source": "steganography",
                "data": {
                    "resonance_type": "mother_sigil",
                    "core_frequencies": stego_data.get("frequencies", []),
                    "signature": stego_data.get("signature", ""),
                    "complete": False
                }
            }
            
        # If both methods failed, return error
        return {
            "source": "none",
            "error": "No resonance data found in the image",
            "complete": False
        }
    
    def generate_sound_parameters(self, resonance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts resonance data into sound synthesis parameters.
        
        Args:
            resonance_data: Resonance data dictionary
            
        Returns:
            Dictionary with sound synthesis parameters
        """
        # Extract the core frequencies
        if isinstance(resonance_data, dict) and "data" in resonance_data:
            core_data = resonance_data["data"]
        else:
            core_data = resonance_data
            
        # Check if we have the required data
        if not "core_frequencies" in core_data:
            return {"error": "Missing core frequencies in resonance data"}
            
        frequencies = core_data["core_frequencies"]
        
        # Generate sound parameters based on the frequencies
        harmonic_ratio = core_data.get("cycle_structures", {}).get("harmonic_ratio", 1.618)
        
        # Build synthesis parameters
        synthesis_params = {
            "carrier_frequencies": frequencies,
            "modulation_indices": [harmonic_ratio * i for i in range(1, len(frequencies) + 1)],
            "envelope": {
                "attack": 0.1,
                "decay": 0.3,
                "sustain": 0.7,
                "release": 2.0
            },
            "spatial": {
                "width": 0.8,
                "depth": 0.6
            }
        }
        
        # Add color-to-sound mappings if colors are available
        if "color_codes" in core_data:
            color_map = {}
            for name, color in core_data["color_codes"].items():
                # Strip the # from hex color
                hex_color = color.lstrip("#")
                # Convert hex to RGB values
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                # Map RGB to audio parameters (simplified mapping)
                color_map[name] = {
                    "filter_cutoff": r / 255.0 * 10000 + 200,  # 200-10200 Hz
                    "resonance": g / 255.0 * 10 + 1,  # Q of 1-11
                    "intensity": b / 255.0  # 0.0-1.0
                }
            synthesis_params["color_mapping"] = color_map
            
        return synthesis_params

if __name__ == "__main__":
    print("Decoder started...")
    decoder = SigilDecoder()
    test_image_path = "glyphs/glyph_resonance/encoded_glyphs/encoded_mother_sigil.jpeg"
    
    resonance = decoder.read_mother_resonance(test_image_path)
    if resonance.get("error"):
        print("Error:", resonance["error"])
    else:
        print("Resonance data:", resonance)
    
    synthesis = decoder.generate_sound_parameters(resonance)
    print("Synthesis Parameters:", synthesis)

