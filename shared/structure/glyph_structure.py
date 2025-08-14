# personal glyphs, spiritual glyphs, semantic glyphs only created when something is meaningful and it is not repeated
glyph_types = {
    "glyph_types_id": "UUID",
    "glyph_type_title": "bool",
    "has_platonic": "bool",
    "has_sacred_geometry": "bool",
    "has_sigil": "bool",
}

glyph_colours = {
    "glyph_colour_id": "UUID",
    "glyph_colour_title": "bool",
    "glyph_colour_hex": "", # hex value
}

sacred_geometry = {
    "sacred_geometry_id": "UUID",
    "sacred_geometry_title": "str",
    "sacred_geometry_colour": "", # hex colour for lines
}

platonics = {
    "platonic_id": "UUID",
    "platonic_title": "str",
    "platonic_colour": "", # hex colour for lines
}

sigils = {
    "sigil_id": "UUID",
    "sigil_title": "str",
    "sigil_category": "",
    "unicode_character": "",
    "sigil_colour": "", # hex colour for symbol
}

glyph_drawing = {
    "glyph_drawing_id": "UUID",
    "glyph_title": "",
    "glyph_image_path": "",
    "fk_glyph_types_id": "UUID",
    "fk_sacred_geometry_id": "UUID",
    "fk_platonic_id": "UUID",
    "fk_sigil_id": "UUID",
}

glyphs = {
    "glyph_id": "UUID",
    "glyph_flag": "bool",
    "glyph_exif_decodings": "", 
    "glyph_steganography_decodings": "", 
    "glyph_exif_encodings": "",
    "glyph_steganography_encodings": "",
    "glyph_tags": "",
    "glyph_flag": "bool", # trigger to create glyph
    "fk_glyph_image_path": "",
    "fk_glyph_types_id": "",
    "fk_sacred_geometry_id": "",
    "fk_platonic_id": "",
    "fk_glyph_sigil": "",
}