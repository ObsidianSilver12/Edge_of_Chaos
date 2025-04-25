from PIL import Image
import piexif
import json
import numpy as np
import os
from .utils import calculate_resonance_signature

def encode_glyph(image_path, resonance_data, output_path=None):
    """
    Encode resonance data into an image file.
    
    Args:
        image_path: Path to source image
        resonance_data: Dictionary with resonance data
        output_path: Where to save encoded image
        
    Returns:
        Path to the encoded image
    """
    # Ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    else:
        # Default to encoded_glyphs folder with same filename
        filename = os.path.basename(image_path)
        output_path = f"src/glyphs/glyph_resonance/encoded_glyphs/{filename}"
        os.makedirs("src/glyphs/glyph_resonance/encoded_glyphs", exist_ok=True)
    
    # Add signature to resonance data
    resonance_data["signature"] = calculate_resonance_signature(resonance_data)
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert resonance data to JSON string
        json_data = json.dumps(resonance_data)
        
        # Create EXIF data with our JSON
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
        exif_dict["0th"][piexif.ImageIFD.Software] = "GlyphResonanceSystem"
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = json_data
        
        # Convert to bytes
        exif_bytes = piexif.dump(exif_dict)
        
        # Save with EXIF data
        img.save(output_path, "JPEG", exif=exif_bytes, quality=95)
        
        print(f"Glyph encoded successfully with EXIF data")
        return output_path
        
    except Exception as e:
        print(f"Error encoding glyph: {e}")
        return None

