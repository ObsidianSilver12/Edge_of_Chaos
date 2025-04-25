from src.glyph_resonance import encode_mother_sigil, read_mother_sigil, is_aeri_sigil, get_sound_parameters
import json

# Path to the Mother Sigil image
image_path = "src/glyph_resonance/glyphs/mother_sigil.jpeg"

# Define custom resonance data (optional)
custom_resonance = {
    "resonance_type": "mother_sigil",
    "core_frequencies": [432.0, 528.0, 639.0, 741.0, 852.0, 963.0],
    "color_codes": {
        "primary": "#4b0082",  # Indigo
        "secondary": "#1e90ff",  # Blue
        "tertiary": "#ffd700"   # Gold
    },
    "cycle_structures": {
        "phases": 7,
        "harmonic_ratio": 1.618,
        "resonance_pattern": "fibonacci"
    },
    "timestamp": "2023-10-15T12:00:00Z"
}

# Encode the mother sigil
encoded_path = encode_mother_sigil(
    image_path=image_path,
    resonance_data=custom_resonance,
    output_path="encoded_mother_sigil.jpeg"
)
print(f"Encoded sigil saved to: {encoded_path}")

# Verify the image is an Aeri sigil
if is_aeri_sigil(encoded_path):
    print("The image is an authentic Aeri sigil")
else:
    print("The image is not an authentic Aeri sigil")

# Read the resonance data
resonance_data = read_mother_sigil(encoded_path)
print("Extracted resonance data:")
print(json.dumps(resonance_data, indent=2))

# Get sound synthesis parameters
sound_params = get_sound_parameters(encoded_path)
print("Sound synthesis parameters:")
print(json.dumps(sound_params, indent=2))
