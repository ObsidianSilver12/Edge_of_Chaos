import json
import piexif
from PIL import Image

def encode_glyph(image_path: str, resonance_data: dict, output_path: str) -> str:
    """
    Encodes a glyph image with resonance data by inserting EXIF metadata.
    """
    # Load image using Pillow
    img = Image.open(image_path)
    
    # Create a baseline EXIF dictionary
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Set the Software tag in the 0th IFD so the decoder can identify our sigil.
    # Tag 305 corresponds to "Software".
    exif_dict["0th"][305] = b'AeriSigilSystem'
    
    # Convert the resonance_data to JSON then to bytes; use tag 37510 (UserComment) in the Exif IFD.
    json_data = json.dumps(resonance_data)
    user_comment_bytes = json_data.encode('utf-8')
    exif_dict["Exif"][37510] = user_comment_bytes
    
    # Dump the EXIF dictionary to bytes
    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception as dump_err:
        raise ValueError(f"Error encoding EXIF data: {dump_err}") from dump_err

    # Save the image with the updated EXIF data
    try:
        img.save(output_path, exif=exif_bytes)
    except Exception as save_err:
        raise ValueError(f"Error saving encoded glyph: {save_err}") from save_err

    return output_path

