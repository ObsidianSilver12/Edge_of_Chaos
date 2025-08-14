"""
Sensory Data Enums
Enumerations for all categorical fields in the sensory data dictionary.
"""
from enum import Enum


# =============================================================================
# VISUAL ENUMS
# =============================================================================

class VisualType(Enum):
    PHOTOGRAPH = "photograph"
    ILLUSTRATION = "illustration"
    VISUALIZATION = "visualization"
    DREAM = "dream"
    HALLUCINATION = "hallucination"
    MEMORY = "memory"
    IMAGINATION = "imagination"
    SCREEN_CAPTURE = "screen_capture"

class ImageSource(Enum):
    CAMERA = "camera"
    GENERATED = "generated"
    MEMORY = "memory"
    IMAGINATION = "imagination"
    SCREEN = "screen"
    UPLOADED = "uploaded"
    PROCESSED = "processed"

class FileFormat(Enum):
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    BMP = ".bmp"
    GIF = ".gif"
    TIFF = ".tiff"
    SVG = ".svg"

class LightingType(Enum):
    NATURAL = "natural"
    ARTIFICIAL = "artificial"
    MIXED = "mixed"
    BACKLIT = "backlit"
    STUDIO = "studio"
    AMBIENT = "ambient"

class CompositionType(Enum):
    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    SYMMETRICAL = "symmetrical"
    ASYMMETRICAL = "asymmetrical"
    CENTERED = "centered"
    DYNAMIC = "dynamic"

class LineType(Enum):
    STRAIGHT = "straight"
    CURVED = "curved"
    DIAGONAL = "diagonal"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ZIGZAG = "zigzag"

class SymmetryType(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    RADIAL = "radial"
    BILATERAL = "bilateral"
    ROTATIONAL = "rotational"
    NONE = "none"

class SceneType(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    ABSTRACT = "abstract"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    STILL_LIFE = "still_life"

# =============================================================================
# AUDITORY ENUMS
# =============================================================================

class AudioSource(Enum):
    SPEECH = "speech"
    MUSIC = "music"
    ENVIRONMENTAL = "environmental"
    MIXED = "mixed"
    SYNTHETIC = "synthetic"
    SILENCE = "silence"

class AudioFormat(Enum):
    WAV = ".wav"
    MP3 = ".mp3"
    FLAC = ".flac"
    AAC = ".aac"
    OGG = ".ogg"
    M4A = ".m4a"

class AudioChannels(Enum):
    MONO = "mono"
    STEREO = "stereo"
    SURROUND_5_1 = "surround_5_1"
    SURROUND_7_1 = "surround_7_1"
    QUAD = "quad"

class VoiceQuality(Enum):
    CLEAR = "clear"
    MUFFLED = "muffled"
    DISTORTED = "distorted"
    WHISPERED = "whispered"
    SHOUTED = "shouted"
    ROBOTIC = "robotic"

class AudioType(Enum):
    SPEECH = "speech"
    MUSIC = "music"
    NOISE = "noise"
    MIXED = "mixed"
    TONE = "tone"
    SILENCE = "silence"

# =============================================================================
# TEXT ENUMS
# =============================================================================

class InputType(Enum):
    TYPED = "typed"
    SPEECH_TO_TEXT = "speech_to_text"
    OCR = "ocr"
    GENERATED = "generated"
    DICTATED = "dictated"
    TRANSCRIBED = "transcribed"

class ContentStructure(Enum):
    PARAGRAPH = "paragraph"
    DIALOGUE = "dialogue"
    LIST = "list"
    BULLET_POINTS = "bullet_points"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"
    POETRY = "poetry"

class LanguageRegister(Enum):
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    COLLOQUIAL = "colloquial"
    SLANG = "slang"

class CommunicationDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INTERNAL = "internal"
    BIDIRECTIONAL = "bidirectional"

class EmotionalTone(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"

# =============================================================================
# EMOTIONAL STATE ENUMS
# =============================================================================

class PrimaryEmotion(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    CONTEMPT = "contempt"

class EmotionalColor(Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    WHITE = "white"
    BLACK = "black"
    GOLD = "gold"

class EmotionalTexture(Enum):
    SMOOTH = "smooth"
    ROUGH = "rough"
    WARM = "warm"
    COLD = "cold"
    SHARP = "sharp"
    SOFT = "soft"
    HEAVY = "heavy"
    LIGHT = "light"

# =============================================================================
# SPATIAL ENUMS
# =============================================================================

class ReferenceFrame(Enum):
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    RELATIVE = "relative"
    BODY_CENTERED = "body_centered"
    WORLD_CENTERED = "world_centered"

class MovementType(Enum):
    LINEAR = "linear"
    ROTATIONAL = "rotational"
    OSCILLATORY = "oscillatory"
    RANDOM = "random"
    CIRCULAR = "circular"
    SPIRAL = "spiral"

# =============================================================================
# TEMPORAL ENUMS
# =============================================================================

class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"
    CYCLICAL = "cyclical"
    RANDOM = "random"

class TemporalPattern(Enum):
    REGULAR = "regular"
    IRREGULAR = "irregular"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    PULSED = "pulsed"
    CONTINUOUS = "continuous"

# =============================================================================
# METAPHYSICAL ENUMS
# =============================================================================

class ExperienceType(Enum):
    VISION = "vision"
    KNOWING = "knowing"
    SENSING = "sensing"
    HEARING = "hearing"
    FEELING = "feeling"
    COMMUNICATION = "communication"
    PRESENCE = "presence"

class ConnectionQuality(Enum):
    CLEAR = "clear"
    FUZZY = "fuzzy"
    STRONG = "strong"
    WEAK = "weak"
    INTERMITTENT = "intermittent"
    STABLE = "stable"

class AwarenessQuality(Enum):
    FOCUSED = "focused"
    DIFFUSE = "diffuse"
    EXPANDED = "expanded"
    CONTRACTED = "contracted"
    HEIGHTENED = "heightened"
    NORMAL = "normal"

class AttentionFocus(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNIFIED = "unified"
    SCATTERED = "scattered"
    CONCENTRATED = "concentrated"
    FLOWING = "flowing"

class DimensionalSense(Enum):
    THIRD_DIMENSION = "3d"
    FOURTH_DIMENSION = "4d"
    FIFTH_DIMENSION = "5d"
    MULTI_DIMENSIONAL = "multi_dimensional"
    TIMELESS = "timeless"
    SPACELESS = "spaceless"

class ProtectionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PARTIAL = "partial"
    COMPROMISED = "compromised"
    STRENGTHENING = "strengthening"
    UNKNOWN = "unknown"

class EnergyQuality(Enum):
    CALM = "calm"
    EXCITED = "excited"
    SCATTERED = "scattered"
    FOCUSED = "focused"
    HEAVY = "heavy"
    LIGHT = "light"
    CHAOTIC = "chaotic"
    HARMONIOUS = "harmonious"

# =============================================================================
# ALGORITHMIC ENUMS
# =============================================================================

class AlgorithmType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NEURAL_NETWORK = "neural_network"
    DECISION_TREE = "decision_tree"
    PATTERN_MATCHING = "pattern_matching"
    OPTIMIZATION = "optimization"

class PerformanceTrend(Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"
    LEARNING = "learning"
    PLATEAUED = "plateaued"

# =============================================================================
# PROCESSING STATUS ENUMS
# =============================================================================

class ProcessingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"

class DataSource(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    GENERATED = "generated"
    MEMORY = "memory"
    CALCULATION = "calculation"
    SENSOR = "sensor"

# =============================================================================
# QUALITY AND CONFIDENCE ENUMS
# =============================================================================

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CERTAIN = "certain"

class QualityLevel(Enum):
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"
    PERFECT = "perfect"

# =============================================================================
# INTENSITY LEVELS
# =============================================================================

class IntensityLevel(Enum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"
    OVERWHELMING = "overwhelming"

# =============================================================================
# ENUM MAPPING DICTIONARY
# =============================================================================

SENSORY_ENUMS = {
    # Visual
    'visual_type': VisualType,
    'image_source': ImageSource,
    'image_format': FileFormat,
    'lighting_type': LightingType,
    'composition_type': CompositionType,
    'line_types': LineType,
    'symmetry_detected': SymmetryType,
    'scene_type': SceneType,
    
    # Auditory
    'audio_source': AudioSource,
    'file_format': AudioFormat,
    'channels': AudioChannels,
    'voice_quality': VoiceQuality,
    'audio_type': AudioType,
    
    # Text
    'input_type': InputType,
    'content_structure': ContentStructure,
    'language_register': LanguageRegister,
    'communication_direction': CommunicationDirection,
    'emotional_tone': EmotionalTone,
    
    # Emotional State
    'primary_emotion': PrimaryEmotion,
    'emotional_color': EmotionalColor,
    'emotional_texture': EmotionalTexture,
    
    # Spatial
    'reference_frame': ReferenceFrame,
    'movement_type': MovementType,
    
    # Temporal
    'trend_direction': TrendDirection,
    'temporal_pattern': TemporalPattern,
    
    # Metaphysical
    'experience_type': ExperienceType,
    'connection_quality': ConnectionQuality,
    'awareness_quality': AwarenessQuality,
    'attention_focus': AttentionFocus,
    'dimensional_sense': DimensionalSense,
    'protection_status': ProtectionStatus,
    'energy_quality': EnergyQuality,
    
    # Algorithmic
    'algorithm_type': AlgorithmType,
    'performance_trend': PerformanceTrend,
    
    # Processing
    'processing_status': ProcessingStatus,
    'source': DataSource,
    
    # Quality/Confidence
    'confidence_level': ConfidenceLevel,
    'quality_level': QualityLevel,
    'intensity_level': IntensityLevel,
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_enum_value(field_name: str, value: str) -> bool:
    """Validate if a value is valid for the given enum field."""
    if field_name in SENSORY_ENUMS:
        enum_class = SENSORY_ENUMS[field_name]
        try:
            enum_class(value)
            return True
        except ValueError:
            return False
    return True  # If no enum defined, accept any value

def get_valid_values(field_name: str) -> list:
    """Get list of valid values for an enum field."""
    if field_name in SENSORY_ENUMS:
        enum_class = SENSORY_ENUMS[field_name]
        return [e.value for e in enum_class]
    return []

def convert_to_enum(field_name: str, value: str):
    """Convert string value to enum instance."""
    if field_name in SENSORY_ENUMS:
        enum_class = SENSORY_ENUMS[field_name]
        try:
            return enum_class(value)
        except ValueError as exc:
            raise ValueError(f"Invalid value '{value}' for field '{field_name}'. "
                             f"Valid values: {get_valid_values(field_name)}") from exc
    return value

# USAGE EXAMPLE
# Validate sensory data
# from sensory_enums import validate_enum_value, convert_to_enum

# # Example usage in sensory processing
# visual_data = {
#     'visual_type': 'photograph',  # Will be validated against VisualType enum
#     'scene_type': 'outdoor',      # Will be validated against SceneType enum
#     'lighting_type': 'natural'    # Will be validated against LightingType enum
# }

# # Validation
# for field, value in visual_data.items():
#     if not validate_enum_value(field, value):
#         print(f"Invalid value '{value}' for field '{field}'")
#     else:
#         enum_value = convert_to_enum(field, value)
#         print(f"✅ {field}: {enum_value}")