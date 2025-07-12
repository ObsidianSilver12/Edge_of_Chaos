import os
import datetime
from glyphs.glyph_resonance.encoder import encode_glyph  # Import encode_sigil function
from stage_1.evolve.core.mother_resonance import create_mother_resonance_data

def run_encode():
    """
    Run the mother sigil encoding process using the sophisticated
    resonance functionality from mother_resonance.py.
    
    Returns:
        Path to the encoded mother sigil
    """
    # Get sophisticated mother resonance data
    mother_data = create_mother_resonance_data()
    
    # Add timestamp
    mother_data["encoding_timestamp"] = datetime.datetime.now().isoformat()
    
    # Source and destination paths
    source_path = "glyphs/glyph_resonance/glyphs/mother_sigil.jpeg"
    output_path = "glyphs/glyph_resonance/encoded_glyphs/encoded_mother_sigil.jpeg"
    
    # Ensure encoded_glyphs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Encode the sigil
    encoded_path = encode_glyph(  # Use encode_sigil function
        image_path=source_path,
        resonance_data=mother_data,
        output_path=output_path
    )
    
    print(f"Mother sigil encoded with full resonance profile and saved to: {encoded_path}")
    return encoded_path

# Allow direct execution of this module
if __name__ == "__main__":
    run_encode()



