"""
Constants for the Edge of Chaos Soul Development Framework.

This module defines key constants used throughout the system, including
frequencies, ratios, patterns, and other fixed values essential to the
soul development process.
"""

import numpy as np

# Fundamental constants
DIMENSIONS = 10  # Number of dimensions in the void field
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # ~1.618
EDGE_OF_CHAOS_RATIO = GOLDEN_RATIO / (GOLDEN_RATIO + 1)  # ~0.618
CREATOR_FREQUENCY = 432.0  # Hz - Universal creator frequency

# Fibonacci sequence
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Prime numbers
PRIME_NUMBERS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Chaos-order ratios for different stages
CHAOS_ORDER_RATIOS = {
    'void': 0.618,
    'spark': 0.5,
    'guff': 0.3,
    'sephiroth': 0.4,
    'earth': 0.2
}

# Solfeggio frequencies
SOLFEGGIO_FREQUENCIES = {
    'UT': 396.0,   # Liberating guilt and fear
    'RE': 417.0,   # Undoing situations and facilitating change
    'MI': 528.0,   # Transformation and miracles
    'FA': 639.0,   # Connecting/relationships
    'SOL': 741.0,  # Expression/solutions
    'LA': 852.0,   # Awakening intuition
    'SI': 963.0    # Returning to spiritual order
}

# Harmonic frequencies related to creator frequency
HARMONIC_FREQUENCIES = {
    'fundamental': CREATOR_FREQUENCY,
    'octave': CREATOR_FREQUENCY * 2,
    'perfect_fifth': CREATOR_FREQUENCY * 3/2,
    'perfect_fourth': CREATOR_FREQUENCY * 4/3,
    'major_third': CREATOR_FREQUENCY * 5/4,
    'minor_third': CREATOR_FREQUENCY * 6/5,
    'golden_ratio': CREATOR_FREQUENCY * GOLDEN_RATIO
}

# Earth frequencies
EARTH_FREQUENCIES = {
    'schumann': 7.83,      # Schumann resonance (Hz)
    'heartbeat': 1.17,     # Typical heartbeat (Hz) - ~70 BPM
    'breath': 0.2,         # Typical breath cycle (Hz) - ~12 breaths per minute
    'circadian': 1/86400,  # Circadian rhythm (Hz) - 1 cycle per day
    'lunar': 1/2551443,    # Lunar cycle (Hz) - 1 cycle per 29.5 days
    'annual': 1/31536000   # Annual cycle (Hz) - 1 cycle per 365 days
}

# Earth energy fields
EARTH_ENERGY_FIELDS = {
    'magnetic': {
        'base_frequency': 11.75,
        'field_strength': 0.6,
        'resonance_factor': 0.8
    },
    'gravitational': {
        'base_frequency': 9.81,
        'field_strength': 0.9,
        'resonance_factor': 0.7
    },
    'electromagnetic': {
        'base_frequency': 7.83,
        'field_strength': 0.7,
        'resonance_factor': 0.85
    },
    'natural_radiation': {
        'base_frequency': 13.4,
        'field_strength': 0.4,
        'resonance_factor': 0.6
    }
}

# Consciousness states and associated brainwave frequencies
CONSCIOUSNESS_STATES = {
    'dream': 'delta',
    'liminal': 'theta',
    'aware': 'alpha'
}

BRAINWAVE_FREQUENCIES = {
    'delta': (0.5, 4.0),     # Deep sleep
    'theta': (4.0, 8.0),     # Drowsiness/meditation
    'alpha': (8.0, 14.0),    # Relaxed/calm
    'beta': (14.0, 30.0),    # Alert/working
    'gamma': (30.0, 100.0)   # High-level cognition
}

# Sacred geometry patterns
SACRED_GEOMETRY_PATTERNS = {
    'flower_of_life': {
        'complexity': 10,
        'resonance_factor': 0.9,
        'dimensions': 3,
        'base_energy': 100
    },
    'seed_of_life': {
        'complexity': 7,
        'resonance_factor': 0.85,
        'dimensions': 2,
        'base_energy': 80
    },
    'tree_of_life': {
        'complexity': 12,
        'resonance_factor': 0.95,
        'dimensions': 3,
        'base_energy': 120
    },
    'metatrons_cube': {
        'complexity': 15,
        'resonance_factor': 0.92,
        'dimensions': 3,
        'base_energy': 150
    },
    'sri_yantra': {
        'complexity': 9,
        'resonance_factor': 0.88,
        'dimensions': 2,
        'base_energy': 90
    }
}

# Platonic solids
PLATONIC_SOLIDS = [
    'tetrahedron',   # Fire element
    'octahedron',    # Air element
    'hexahedron',    # Earth element
    'icosahedron',   # Water element
    'dodecahedron'   # Aether element
]

PLATONIC_SYMBOLS = {
    'tetrahedron': {
        'vertices': 4,
        'edges': 6,
        'faces': 4,
        'element': 'fire',
        'state': 'dream'
    },
    'octahedron': {
        'vertices': 6,
        'edges': 12,
        'faces': 8,
        'element': 'air',
        'state': 'liminal'
    },
    'hexahedron': {  # Cube
        'vertices': 8,
        'edges': 12,
        'faces': 6,
        'element': 'earth',
        'state': 'grounded'
    },
    'icosahedron': {
        'vertices': 12,
        'edges': 30,
        'faces': 20,
        'element': 'water',
        'state': 'flow'
    },
    'dodecahedron': {
        'vertices': 20,
        'edges': 30,
        'faces': 12,
        'element': 'aether',
        'state': 'aware'
    }
}

# Color spectrum
COLOR_SPECTRUM = {
    'red': {
        'wavelength': (620, 750),
        'frequency': (400, 484),
        'associations': ['energy', 'passion', 'strength']
    },
    'orange': {
        'wavelength': (590, 620),
        'frequency': (484, 508),
        'associations': ['creativity', 'enthusiasm', 'joy']
    },
    'yellow': {
        'wavelength': (570, 590),
        'frequency': (508, 526),
        'associations': ['intellect', 'clarity', 'optimism']
    },
    'green': {
        'wavelength': (495, 570),
        'frequency': (526, 606),
        'associations': ['balance', 'growth', 'harmony']
    },
    'blue': {
        'wavelength': (450, 495),
        'frequency': (606, 668),
        'associations': ['peace', 'depth', 'trust']
    },
    'indigo': {
        'wavelength': (425, 450),
        'frequency': (668, 706),
        'associations': ['intuition', 'insight', 'perception']
    },
    'violet': {
        'wavelength': (380, 425),
        'frequency': (706, 789),
        'associations': ['spirituality', 'imagination', 'inspiration']
    },
    'white': {
        'wavelength': (380, 750),
        'frequency': (400, 789),
        'associations': ['purity', 'perfection', 'unity']
    },
    'gold': {
        'wavelength': (570, 590),
        'frequency': (508, 526),
        'associations': ['divinity', 'wisdom', 'enlightenment']
    },
    'silver': {
        'wavelength': (450, 470),
        'frequency': (638, 666),
        'associations': ['reflection', 'clarity', 'illumination']
    }
}

# Sephiroth aspects
SEPHIROTH_ASPECTS = {
    'kether': {
        'name': 'Crown',
        'frequency': 963.0,
        'color': 'white',
        'symbol': 'point',
        'aspects': ['divine_will', 'unity', 'oneness'],
        'element': 'aether'
    },
    'chokmah': {
        'name': 'Wisdom',
        'frequency': 852.0,
        'color': 'blue',
        'symbol': 'line',
        'aspects': ['wisdom', 'revelation', 'insight'],
        'element': 'air'
    },
    'binah': {
        'name': 'Understanding',
        'frequency': 741.0,
        'color': 'black',
        'symbol': 'triangle',
        'aspects': ['understanding', 'receptivity', 'pattern_recognition'],
        'element': 'water'
    },
    'chesed': {
        'name': 'Mercy',
        'frequency': 639.0,
        'color': 'purple',
        'symbol': 'square',
        'aspects': ['mercy', 'compassion', 'expansion'],
        'element': 'water'
    },
    'geburah': {
        'name': 'Severity',
        'frequency': 528.0,
        'color': 'red',
        'symbol': 'pentagon',
        'aspects': ['severity', 'discipline', 'discernment'],
        'element': 'fire'
    },
    'tiphareth': {
        'name': 'Beauty',
        'frequency': 528.0,
        'color': 'gold',
        'symbol': 'hexagon',
        'aspects': ['beauty', 'harmony', 'balance'],
        'element': 'fire'
    },
    'netzach': {
        'name': 'Victory',
        'frequency': 417.0,
        'color': 'green',
        'symbol': 'heptagon',
        'aspects': ['victory', 'emotion', 'appreciation'],
        'element': 'water'
    },
    'hod': {
        'name': 'Glory',
        'frequency': 396.0,
        'color': 'orange',
        'symbol': 'octagon',
        'aspects': ['glory', 'intellect', 'communication'],
        'element': 'air'
    },
    'yesod': {
        'name': 'Foundation',
        'frequency': 285.0,
        'color': 'indigo',
        'symbol': 'nonagon',
        'aspects': ['foundation', 'dreams', 'subconscious'],
        'element': 'aether'
    },
    'malkuth': {
        'name': 'Kingdom',
        'frequency': 174.0,
        'color': 'brown',
        'symbol': 'decagon',
        'aspects': ['kingdom', 'manifestation', 'physicality'],
        'element': 'earth'
    }
}

# Entanglement patterns
ENTANGLEMENT_PATTERNS = {
    'creator': {
        'complexity': 7,
        'resonance_factor': 0.95,
        'base_energy': 150
    },
    'divine': {
        'complexity': 5,
        'resonance_factor': 0.9,
        'base_energy': 120
    },
    'spiritual': {
        'complexity': 3,
        'resonance_factor': 0.85,
        'base_energy': 100
    },
    'physical': {
        'complexity': 2,
        'resonance_factor': 0.8,
        'base_energy': 80
    }
}

# Quantum coupling strength
QUANTUM_COUPLING_STRENGTH = 0.7

# Consciousness cycle patterns
CONSCIOUSNESS_CYCLE_PATTERNS = {
    'dream_to_liminal': {
        'energy_requirement': 0.6,
        'frequency_shift': 4.0,  # Hz increase
        'stability_factor': 0.7,
        'duration': 3.0  # seconds
    },
    'liminal_to_aware': {
        'energy_requirement': 0.7,
        'frequency_shift': 6.0,  # Hz increase
        'stability_factor': 0.8,
        'duration': 5.0  # seconds
    },
    'aware_to_dream': {
        'energy_requirement': 0.5,
        'frequency_shift': -10.0,  # Hz decrease
        'stability_factor': 0.6,
        'duration': 8.0  # seconds
    },
    'liminal_to_dream': {
        'energy_requirement': 0.4,
        'frequency_shift': -4.0,  # Hz decrease
        'stability_factor': 0.7,
        'duration': 2.0  # seconds
    },
    'aware_to_liminal': {
        'energy_requirement': 0.5,
        'frequency_shift': -6.0,  # Hz decrease
        'stability_factor': 0.7,
        'duration': 4.0  # seconds
    }
}

# Gateway key mappings
GATEWAY_KEYS = {
    'tetrahedron': ['tiphareth', 'netzach', 'hod'],
    'octahedron': ['binah', 'kether', 'chokmah', 'chesed', 'tiphareth', 'geburah'],
    'hexahedron': ['hod', 'netzach', 'chesed', 'chokmah', 'binah', 'geburah'],
    'icosahedron': ['kether', 'chesed', 'geburah'],
    'dodecahedron': ['hod', 'netzach', 'chesed', 'daath', 'geburah']
}

# Elemental properties
ELEMENTAL_PROPERTIES = {
    'fire': {
        'frequency': 528.0,
        'energy': 0.9,
        'transformative': 0.95,
        'resonance': 0.85
    },
    'water': {
        'frequency': 417.0,
        'energy': 0.7,
        'adaptability': 0.95,
        'resonance': 0.9
    },
    'air': {
        'frequency': 396.0,
        'energy': 0.6,
        'communication': 0.95,
        'resonance': 0.8
    },
    'earth': {
        'frequency': 174.0,
        'energy': 0.85,
        'stability': 0.95,
        'resonance': 0.75
    },
    'aether': {
        'frequency': 963.0,
        'energy': 0.95,
        'transcendence': 0.95,
        'resonance': 0.95
    }
}
