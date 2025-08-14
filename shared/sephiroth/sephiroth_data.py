# --- START OF FILE sephiroth_data.py ---

"""
Sephiroth Data Definitions

Contains the comprehensive data dictionary defining the properties
and aspects of each of the 11 Sephiroth.
"""

from typing import Dict, List, Any, Tuple
from shared.constants.constants import PHI

# Import shared constants if needed (e.g., for default frequencies)
# from shared.constants.constants import * # Example

# Define base frequencies (can be moved to constants if preferred)
# Example: Use Solfeggio or other meaningful frequencies
KETHER_FREQ = 963.0
CHOKMAH_FREQ = 852.0 # Reversed path? Or wisdom freq?
BINAH_FREQ = 396.0 # Understanding/Grounding?
CHESED_FREQ = 639.0 # Connection/Mercy
GEBURAH_FREQ = 417.0 # Change/Severity
TIPHARETH_FREQ = 528.0 # Love/Beauty/Miracles
NETZACH_FREQ = 741.0 # Intuition/Victory
HOD_FREQ = 741.0 # Awakening/Splendor (same as Netzach? adjust?)
YESOD_FREQ = 852.0 # Spiritual order/Foundation
MALKUTH_FREQ = 174.0 # Lowest Solfeggio - Grounding (if using Solfeggio scale)
DAATH_FREQ = 444.0 # Knowledge - A=444Hz tuning example

# Define the main data structure
SEPHIROTH_ASPECT_DATA: Dict[str, Dict[str, Any]] = {

    # --- Kether ---
    'kether': {
        'name': "Kether", 'title': "Crown", 'base_frequency': KETHER_FREQ,
        'primary_color': 'white', 'element': 'aether/light',
        'geometric_correspondence': 'point', # Source point
        'platonic_affinity': 'sphere', # Contains all potential
        'divine_attribute': "Unity",
        'primary_aspects': ['divine_unity', 'pure_being', 'creation_source'],
        'secondary_aspects': ['divine_will', 'transcendence', 'crown_consciousness'],
        'detailed_aspects': {
            'divine_unity': {'frequency': KETHER_FREQ, 'color': 'brilliant_white', 'element': 'aether', 'keywords': ['oneness', 'unity', 'source'], 'description': 'Absolute divine unity.', 'strength': 1.0},
            'pure_being': {'frequency': KETHER_FREQ*1.1, 'color': 'clear_light', 'element': 'light', 'keywords': ['existence', 'presence', 'is-ness'], 'description': 'The state of pure being.', 'strength': 0.98},
            'creation_source': {'frequency': KETHER_FREQ*0.9, 'color': 'radiant_white', 'element': 'aether', 'keywords': ['potential', 'beginning', 'origin'], 'description': 'The unmanifest source of creation.', 'strength': 0.97},
            'divine_will': {'frequency': KETHER_FREQ*1.05, 'color': 'white_gold', 'element': 'aether', 'keywords': ['intention', 'purpose', 'first_cause'], 'description': 'The primal divine will.', 'strength': 0.95},
            'transcendence': {'frequency': KETHER_FREQ*1.2, 'color': 'pure_light', 'element': 'light', 'keywords': ['beyond', 'limitless', 'infinite'], 'description': 'Transcendence of all form.', 'strength': 0.96},
            'crown_consciousness': {'frequency': KETHER_FREQ, 'color': 'white', 'element': 'light', 'keywords': ['awareness', 'consciousness_source'], 'description': 'The root of all consciousness.', 'strength': 0.99},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.5, 2.0, 3.0, PHI, 5.0], 'falloff': 0.05},
        'stability_modifier': 1.2, 'coherence_modifier': 1.2, 'energy_level': 1.0, 'chaos_order_bias': 0.5, 'resonance_multiplier': 1.1
    },

    # --- Chokmah ---
    'chokmah': {
        'name': "Chokmah", 'title': "Wisdom", 'base_frequency': CHOKMAH_FREQ,
        'primary_color': 'grey', 'element': 'fire/aether', # Primal fire/energy
        'geometric_correspondence': 'line', # First expression/vector
        'platonic_affinity': 'tetrahedron', # Simplest form, fire assoc.
        'divine_attribute': "Wisdom",
        'primary_aspects': ['pure_energy', 'divine_wisdom', 'inspiration'],
        'secondary_aspects': ['dynamic_force', 'primal_masculine'],
        'detailed_aspects': {
            'pure_energy': {'frequency': CHOKMAH_FREQ*1.1, 'color': 'silver', 'element': 'fire', 'keywords': ['force', 'vitality', 'potential'], 'description': 'Undifferentiated divine energy.', 'strength': 0.95},
            'divine_wisdom': {'frequency': CHOKMAH_FREQ, 'color': 'grey', 'element': 'aether', 'keywords': ['insight', 'knowing', 'revelation'], 'description': 'The flash of divine insight.', 'strength': 0.98},
            'inspiration': {'frequency': CHOKMAH_FREQ*PHI, 'color': 'bright_silver', 'element': 'air', 'keywords': ['creativity', 'spark', 'intuition'], 'description': 'The spark of divine inspiration.', 'strength': 0.92},
            'dynamic_force': {'frequency': CHOKMAH_FREQ*0.9, 'color': 'dark_grey', 'element': 'fire', 'keywords': ['power', 'drive', 'momentum'], 'description': 'The driving force of creation.', 'strength': 0.90},
            'primal_masculine': {'frequency': CHOKMAH_FREQ*1.2, 'color': 'grey_blue', 'element': 'fire', 'keywords': ['yang', 'active', 'father'], 'description': 'The active, projective principle.', 'strength': 0.88},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.5, 2.0, 2.5, 3.0, PHI], 'falloff': 0.08},
        'stability_modifier': 1.1, 'coherence_modifier': 1.0, 'energy_level': 0.95, 'chaos_order_bias': 0.55, 'resonance_multiplier': 1.05
    },

    # --- Binah ---
    'binah': {
        'name': "Binah", 'title': "Understanding", 'base_frequency': BINAH_FREQ,
        'primary_color': 'black', 'element': 'water/earth', # Formative waters
        'geometric_correspondence': 'triangle', # First stable form
        'platonic_affinity': 'hexahedron', # Structure, stability, earth assoc.
        'divine_attribute': "Understanding",
        'primary_aspects': ['structure', 'divine_understanding', 'receptivity'],
        'secondary_aspects': ['limitation', 'form', 'primal_feminine'],
        'detailed_aspects': {
            'structure': {'frequency': BINAH_FREQ*0.9, 'color': 'dark_grey', 'element': 'earth', 'keywords': ['form', 'order', 'stability', 'pattern'], 'description': 'The principle of structure and form.', 'strength': 0.92},
            'divine_understanding': {'frequency': BINAH_FREQ, 'color': 'black', 'element': 'water', 'keywords': ['comprehension', 'intelligence', 'reason'], 'description': 'Deep, intuitive understanding.', 'strength': 0.96},
            'receptivity': {'frequency': BINAH_FREQ*1.1, 'color': 'deep_blue', 'element': 'water', 'keywords': ['openness', 'passivity', 'womb'], 'description': 'The receptive principle that gives form.', 'strength': 0.94},
            'limitation': {'frequency': BINAH_FREQ*0.8, 'color': 'grey_black', 'element': 'earth', 'keywords': ['boundary', 'restriction', 'definition'], 'description': 'The principle of limitation and definition.', 'strength': 0.88},
            'form': {'frequency': BINAH_FREQ*0.95, 'color': 'brown_black', 'element': 'earth', 'keywords': ['manifestation', 'shape', 'structure'], 'description': 'The potential for form.', 'strength': 0.90},
            'primal_feminine': {'frequency': BINAH_FREQ*1.05, 'color': 'black_silver', 'element': 'water', 'keywords': ['yin', 'passive', 'mother'], 'description': 'The passive, receptive principle.', 'strength': 0.89},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.333, 1.666, 2.0, 2.666, PHI], 'falloff': 0.1},
        'stability_modifier': 1.15, 'coherence_modifier': 1.1, 'energy_level': 0.90, 'chaos_order_bias': 0.45, 'resonance_multiplier': 1.0
    },

     # --- Chesed ---
    'chesed': {
        'name': "Chesed", 'title': "Mercy", 'base_frequency': CHESED_FREQ,
        'primary_color': 'blue', 'element': 'water', # Flowing kindness
        'geometric_correspondence': 'square', # Stability, structure from mercy
        'platonic_affinity': 'icosahedron', # Water association, complex flow
        'divine_attribute': "Mercy",
        'primary_aspects': ['loving_kindness', 'mercy', 'magnanimity'],
        'secondary_aspects': ['expansion', 'healing', 'benevolence', 'jupiter'],
        'detailed_aspects': {
            'loving_kindness': {'frequency': 528.0, 'color': 'blue', 'element': 'water', 'keywords': ['compassion', 'love', 'kindness', 'healing'], 'description': 'The aspect of unconditional love and compassion.', 'strength': 0.95},
            'mercy': {'frequency': CHESED_FREQ, 'color': 'deep_blue', 'element': 'water', 'keywords': ['forgiveness', 'grace', 'clemency'], 'description': 'The aspect of divine mercy and forgiveness.', 'strength': 0.98},
            'magnanimity': {'frequency': 417.0, 'color': 'royal_blue', 'element': 'water', 'keywords': ['generosity', 'benevolence', 'abundance'], 'description': 'The aspect of generosity and abundance.', 'strength': 0.90},
            'expansion': {'frequency': CHESED_FREQ * 1.1, 'color': 'light_blue', 'element': 'air/water', 'keywords': ['growth', 'abundance', 'prosperity'], 'description': 'The aspect of expansive growth.', 'strength': 0.88},
            'healing': {'frequency': 741.0, 'color': 'turquoise', 'element': 'water', 'keywords': ['restoration', 'wholeness', 'wellness'], 'description': 'The aspect of healing and restoration.', 'strength': 0.85},
            'benevolence': {'frequency': 432.0, 'color': 'sky_blue', 'element': 'water', 'keywords': ['goodwill', 'charity', 'altruism'], 'description': 'The aspect of goodwill and altruism.', 'strength': 0.89},
            'jupiter': {'frequency': 183.58, 'color': 'royal_blue', 'element': 'aether', 'keywords': ['expansion', 'blessing', 'wisdom'], 'description': 'The planetary aspect of Jupiter.', 'strength': 0.80},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.25, 1.5, 2.0, 2.5, 3.0], 'falloff': 0.12},
        'stability_modifier': 1.1, 'coherence_modifier': 1.0, 'energy_level': 0.88, 'chaos_order_bias': 0.40, 'resonance_multiplier': 1.0
    },

     # --- Geburah ---
    'geburah': {
        'name': "Geburah", 'title': "Severity", 'base_frequency': GEBURAH_FREQ,
        'primary_color': 'red', 'element': 'fire', # Judgement, power
        'geometric_correspondence': 'pentagon', # Power, dynamic balance
        'platonic_affinity': 'tetrahedron', # Fire association
        'divine_attribute': "Severity",
        'primary_aspects': ['strength', 'judgement', 'discipline'],
        'secondary_aspects': ['restraint', 'power', 'fear', 'mars'],
        'detailed_aspects': {
            'strength': {'frequency': GEBURAH_FREQ * 1.1, 'color': 'scarlet', 'element': 'fire', 'keywords': ['power', 'might', 'courage'], 'description': 'Focused divine strength and power.', 'strength': 0.96},
            'judgement': {'frequency': GEBURAH_FREQ, 'color': 'red', 'element': 'fire/air', 'keywords': ['discernment', 'justice', 'severity'], 'description': 'The aspect of divine judgement and discernment.', 'strength': 0.94},
            'discipline': {'frequency': GEBURAH_FREQ * 0.9, 'color': 'crimson', 'element': 'earth/fire', 'keywords': ['control', 'restraint', 'focus'], 'description': 'The aspect of self-discipline and control.', 'strength': 0.90},
            'restraint': {'frequency': GEBURAH_FREQ * 0.8, 'color': 'dark_red', 'element': 'earth', 'keywords': ['limitation', 'boundary', 'holding_back'], 'description': 'The power of necessary restraint.', 'strength': 0.85},
            'power': {'frequency': GEBURAH_FREQ * 1.2, 'color': 'bright_red', 'element': 'fire', 'keywords': ['might', 'authority', 'force'], 'description': 'Raw divine power.', 'strength': 0.92},
            'fear': {'frequency': GEBURAH_FREQ * 0.7, 'color': 'blood_red', 'element': 'shadow', 'keywords': ['awe', 'respect', 'dread'], 'description': 'The awe or fear inspired by divine power (potential negative).', 'strength': 0.70}, # Shadow aspect
            'mars': {'frequency': 144.72, 'color': 'red', 'element': 'fire', 'keywords': ['action', 'energy', 'conflict'], 'description': 'The planetary aspect of Mars.', 'strength': 0.82},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.2, 1.44, 1.6, 2.0, 2.4], 'falloff': 0.15}, # More dissonant?
        'stability_modifier': 1.0, 'coherence_modifier': 0.9, 'energy_level': 0.85, 'chaos_order_bias': 0.60, 'resonance_multiplier': 0.95
    },

    # --- Tiphareth ---
    'tiphareth': {
        'name': "Tiphareth", 'title': "Beauty", 'base_frequency': TIPHARETH_FREQ,
        'primary_color': 'yellow', 'element': 'air/fire', # Solar radiance
        'geometric_correspondence': 'hexagram', # Balance of opposites
        'platonic_affinity': 'octahedron', # Air association, balance
        'divine_attribute': "Beauty",
        'primary_aspects': ['harmony', 'balance', 'compassion', 'beauty'],
        'secondary_aspects': ['healing', 'integration', 'sacrifice', 'sun'],
        'detailed_aspects': {
            'harmony': {'frequency': TIPHARETH_FREQ * PHI, 'color': 'gold', 'element': 'air', 'keywords': ['balance', 'synthesis', 'peace'], 'description': 'The principle of harmony and balance.', 'strength': 0.98},
            'balance': {'frequency': TIPHARETH_FREQ, 'color': 'yellow', 'element': 'air', 'keywords': ['equilibrium', 'center', 'symmetry'], 'description': 'The central point of balance.', 'strength': 0.96},
            'compassion': {'frequency': 528.0, 'color': 'yellow_green', 'element': 'water/air', 'keywords': ['empathy', 'love', 'understanding'], 'description': 'Compassion born from balance.', 'strength': 0.94},
            'beauty': {'frequency': TIPHARETH_FREQ * 1.1, 'color': 'bright_yellow', 'element': 'fire/light', 'keywords': ['aesthetics', 'radiance', 'splendor'], 'description': 'The aspect of divine beauty.', 'strength': 0.97},
            'healing': {'frequency': TIPHARETH_FREQ * 1.2, 'color': 'golden_yellow', 'element': 'light', 'keywords': ['restoration', 'wholeness', 'integration'], 'description': 'The integration and healing aspect.', 'strength': 0.90},
            'integration': {'frequency': TIPHARETH_FREQ * 0.9, 'color': 'amber', 'element': 'earth/air', 'keywords': ['synthesis', 'unity', 'completion'], 'description': 'The integration of opposites.', 'strength': 0.88},
            'sacrifice': {'frequency': TIPHARETH_FREQ * 0.8, 'color': 'deep_gold', 'element': 'fire', 'keywords': ['selflessness', 'offering', 'transmutation'], 'description': 'The concept of sacrifice for higher purpose.', 'strength': 0.85},
            'sun': {'frequency': 126.22, 'color': 'gold', 'element': 'fire', 'keywords': ['consciousness', 'vitality', 'center'], 'description': 'The planetary aspect of the Sun.', 'strength': 0.92},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.5, 2.0, PHI, 2.5, 3.0, PHI*2], 'falloff': 0.08},
        'stability_modifier': 1.1, 'coherence_modifier': 1.15, 'energy_level': 0.92, 'chaos_order_bias': 0.50, 'resonance_multiplier': 1.1
    },

     # --- Netzach ---
    'netzach': {
        'name': "Netzach", 'title': "Victory", 'base_frequency': NETZACH_FREQ,
        'primary_color': 'green', 'element': 'fire/air', # Emotional fire, air of inspiration
        'geometric_correspondence': 'heptagon', # 7 emotions, victory
        'platonic_affinity': 'tetrahedron', # Fire association
        'divine_attribute': "Victory",
        'primary_aspects': ['endurance', 'inspiration', 'desire', 'victory'],
        'secondary_aspects': ['emotion', 'instinct', 'venus'],
        'detailed_aspects': {
            'endurance': {'frequency': NETZACH_FREQ*0.9, 'color': 'emerald', 'element': 'earth/fire', 'keywords': ['persistence', 'stamina', 'fortitude'], 'description': 'The quality of endurance and persistence.', 'strength': 0.90},
            'inspiration': {'frequency': NETZACH_FREQ*1.1, 'color': 'bright_green', 'element': 'air/fire', 'keywords': ['creativity', 'passion', 'artistry'], 'description': 'Artistic and emotional inspiration.', 'strength': 0.94},
            'desire': {'frequency': NETZACH_FREQ, 'color': 'green', 'element': 'fire', 'keywords': ['passion', 'longing', 'drive'], 'description': 'The driving force of desire and passion.', 'strength': 0.92},
            'victory': {'frequency': NETZACH_FREQ*1.2, 'color': 'lime_green', 'element': 'fire', 'keywords': ['triumph', 'success', 'achievement'], 'description': 'The aspect of victory over obstacles.', 'strength': 0.95},
            'emotion': {'frequency': NETZACH_FREQ*0.95, 'color': 'olive_green', 'element': 'water/fire', 'keywords': ['feeling', 'passion', 'sensitivity'], 'description': 'The realm of raw emotion and feeling.', 'strength': 0.88},
            'instinct': {'frequency': NETZACH_FREQ*0.85, 'color': 'forest_green', 'element': 'earth/fire', 'keywords': ['intuition', 'gut_feeling', 'drive'], 'description': 'Primal instincts and drives.', 'strength': 0.86},
            'venus': {'frequency': 221.23, 'color': 'green', 'element': 'earth/water', 'keywords': ['love', 'beauty', 'harmony', 'art'], 'description': 'The planetary aspect of Venus.', 'strength': 0.89},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, PHI], 'falloff': 0.13},
        'stability_modifier': 0.9, 'coherence_modifier': 0.95, 'energy_level': 0.80, 'chaos_order_bias': 0.65, 'resonance_multiplier': 0.9
    },

    # --- Hod ---
    'hod': {
        'name': "Hod", 'title': "Splendor", 'base_frequency': HOD_FREQ, # Often same base as Netzach
        'primary_color': 'orange', 'element': 'water/air', # Intellect (air) reflecting (water)
        'geometric_correspondence': 'octagon', # Structure, intellect, 8 paths
        'platonic_affinity': 'octahedron', # Air association
        'divine_attribute': "Splendor",
        'primary_aspects': ['intellect', 'communication', 'reason', 'splendor'],
        'secondary_aspects': ['logic', 'structure', 'mercury'],
        'detailed_aspects': {
            'intellect': {'frequency': HOD_FREQ, 'color': 'orange', 'element': 'air', 'keywords': ['mind', 'thought', 'analysis'], 'description': 'The power of the rational intellect.', 'strength': 0.96},
            'communication': {'frequency': HOD_FREQ*1.1, 'color': 'yellow_orange', 'element': 'air', 'keywords': ['language', 'expression', 'connection'], 'description': 'The ability to communicate and express.', 'strength': 0.94},
            'reason': {'frequency': HOD_FREQ*0.9, 'color': 'red_orange', 'element': 'air/earth', 'keywords': ['logic', 'rationality', 'order'], 'description': 'The faculty of reason and logic.', 'strength': 0.90},
            'splendor': {'frequency': HOD_FREQ*1.05, 'color': 'bright_orange', 'element': 'fire/light', 'keywords': ['glory', 'majesty', 'radiance'], 'description': 'The aspect of divine splendor.', 'strength': 0.92},
            'logic': {'frequency': HOD_FREQ*0.95, 'color': 'brown_orange', 'element': 'earth/air', 'keywords': ['structure', 'order', 'system'], 'description': 'Logical structure and systems thinking.', 'strength': 0.88},
            'structure': {'frequency': HOD_FREQ*0.85, 'color': 'dark_orange', 'element': 'earth', 'keywords': ['form', 'pattern', 'organization'], 'description': 'The underlying structure of thought.', 'strength': 0.86},
            'mercury': {'frequency': 141.27, 'color': 'orange', 'element': 'air', 'keywords': ['intellect', 'communication', 'mind'], 'description': 'The planetary aspect of Mercury.', 'strength': 0.87},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.2, 1.333, 1.666, 1.8, 2.0, 2.4], 'falloff': 0.14},
        'stability_modifier': 1.05, 'coherence_modifier': 1.0, 'energy_level': 0.78, 'chaos_order_bias': 0.40, 'resonance_multiplier': 0.95
    },

    # --- Yesod ---
    'yesod': {
        'name': "Yesod", 'title': "Foundation", 'base_frequency': YESOD_FREQ,
        'primary_color': 'violet', 'element': 'water/aether', # Astral waters, foundation
        'geometric_correspondence': 'nonagon', # 9 represents completion before manifestation
        'platonic_affinity': 'icosahedron', # Water association
        'divine_attribute': "Foundation",
        'primary_aspects': ['imagination', 'subconscious', 'foundation', 'reflection'],
        'secondary_aspects': ['memory', 'cycles', 'moon'],
        'detailed_aspects': {
            'imagination': {'frequency': YESOD_FREQ*1.1, 'color': 'violet', 'element': 'aether', 'keywords': ['creativity', 'vision', 'dreams'], 'description': 'The power of imagination and visualization.', 'strength': 0.95},
            'subconscious': {'frequency': YESOD_FREQ*0.9, 'color': 'deep_purple', 'element': 'water', 'keywords': ['unconscious', 'instinct', 'patterns'], 'description': 'The realm of the subconscious mind.', 'strength': 0.90},
            'foundation': {'frequency': YESOD_FREQ, 'color': 'purple', 'element': 'earth/water', 'keywords': ['base', 'support', 'stability'], 'description': 'The foundation upon which reality rests.', 'strength': 0.98},
            'reflection': {'frequency': YESOD_FREQ*1.05, 'color': 'lavender', 'element': 'water/light', 'keywords': ['mirror', 'image', 'astral'], 'description': 'Reflecting higher realities; the astral plane.', 'strength': 0.92},
            'memory': {'frequency': YESOD_FREQ*0.95, 'color': 'dark_violet', 'element': 'water', 'keywords': ['past', 'records', 'akashic'], 'description': 'The repository of memory.', 'strength': 0.88},
            'cycles': {'frequency': YESOD_FREQ*0.85, 'color': 'blue_purple', 'element': 'water', 'keywords': ['rhythm', 'flow', 'tides'], 'description': 'The influence of natural cycles.', 'strength': 0.85},
            'moon': {'frequency': 210.42, 'color': 'silver_purple', 'element': 'water', 'keywords': ['subconscious', 'emotion', 'cycles'], 'description': 'The planetary aspect of the Moon.', 'strength': 0.90},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.125, 1.25, 1.5, 1.875, 2.0, PHI], 'falloff': 0.16},
        'stability_modifier': 1.0, 'coherence_modifier': 1.05, 'energy_level': 0.75, 'chaos_order_bias': 0.45, 'resonance_multiplier': 1.0
    },

    # --- Malkuth ---
    'malkuth': {
        'name': "Malkuth", 'title': "Kingdom", 'base_frequency': MALKUTH_FREQ,
        'primary_color': 'earth_tones', # Citrine, Olive, Russet, Black
        'element': 'earth', # Material realm
        'geometric_correspondence': 'cross/cube', # Manifestation in 4 elements/3D space
        'platonic_affinity': 'hexahedron', # Earth association
        'divine_attribute': "Kingdom",
        'primary_aspects': ['manifestation', 'physicality', 'grounding', 'presence'],
        'secondary_aspects': ['sensory_experience', 'elements', 'earth'],
        'detailed_aspects': {
            'manifestation': {'frequency': MALKUTH_FREQ*1.1, 'color': 'russet', 'element': 'earth', 'keywords': ['reality', 'form', 'embodiment'], 'description': 'The plane of physical manifestation.', 'strength': 0.98},
            'physicality': {'frequency': MALKUTH_FREQ, 'color': 'brown', 'element': 'earth', 'keywords': ['body', 'matter', 'density'], 'description': 'The experience of the physical body and world.', 'strength': 0.95},
            'grounding': {'frequency': MALKUTH_FREQ*0.9, 'color': 'black', 'element': 'earth', 'keywords': ['stability', 'foundation', 'roots'], 'description': 'Connection to the Earth and physical stability.', 'strength': 0.92},
            'presence': {'frequency': MALKUTH_FREQ*1.05, 'color': 'olive', 'element': 'earth', 'keywords': ['here_now', 'awareness', 'being'], 'description': 'Being present in the physical moment.', 'strength': 0.90},
            'sensory_experience': {'frequency': MALKUTH_FREQ*1.2, 'color': 'citrine', 'element': 'earth/air', 'keywords': ['senses', 'perception', 'feeling'], 'description': 'The input from the physical senses.', 'strength': 0.88},
            'elements': {'frequency': MALKUTH_FREQ*0.8, 'color': 'multi_earth', 'element': 'earth', 'keywords': ['fire', 'water', 'air', 'earth'], 'description': 'The interplay of the four classical elements.', 'strength': 0.85},
            'earth': {'frequency': 194.18, 'color': 'green_brown', 'element': 'earth', 'keywords': ['gaia', 'nature', 'life'], 'description': 'The planetary aspect of Earth.', 'strength': 0.94},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.125, 1.25, 1.333, 1.5, 1.666], 'falloff': 0.2}, # Dense, fewer high harmonics
        'stability_modifier': 1.2, 'coherence_modifier': 0.9, 'energy_level': 0.70, 'chaos_order_bias': 0.35, 'resonance_multiplier': 0.9
    },

    # --- Daath ---
    'daath': {
        'name': "Daath", 'title': "Knowledge", 'base_frequency': DAATH_FREQ,
        'primary_color': 'lavender', # Invisible/hidden nature
        'element': 'aether/shadow', # Intersection of spiritual and shadow
        'geometric_correspondence': 'vesica_piscis', # Gateway, intersection
        'platonic_affinity': 'none', # Not a true Sephirah, abyss/gateway
        'divine_attribute': "Knowledge",
        'primary_aspects': ['hidden_knowledge', 'abyss', 'integration', 'ego_death'],
        'secondary_aspects': ['transformation', 'gateway', 'shadow_self'],
        'detailed_aspects': {
            'hidden_knowledge': {'frequency': DAATH_FREQ, 'color': 'grey', 'element': 'aether', 'keywords': ['secrets', 'gnosis', 'occult'], 'description': 'Knowledge hidden in the Abyss.', 'strength': 0.92},
            'abyss': {'frequency': DAATH_FREQ*0.8, 'color': 'black_hole', 'element': 'void/shadow', 'keywords': ['void', 'separation', 'crossing'], 'description': 'The Abyss separating triads.', 'strength': 0.90},
            'integration': {'frequency': DAATH_FREQ*1.1, 'color': 'lavender', 'element': 'aether', 'keywords': ['synthesis', 'unity', 'balance'], 'description': 'Integration of higher and lower self.', 'strength': 0.88},
            'ego_death': {'frequency': DAATH_FREQ*0.9, 'color': 'dark_grey', 'element': 'shadow', 'keywords': ['surrender', 'transformation', 'letting_go'], 'description': 'The necessary dissolution of the ego.', 'strength': 0.85},
            'transformation': {'frequency': 528.0, 'color': 'silver_lavender', 'element': 'aether', 'keywords': ['change', 'alchemy', 'metamorphosis'], 'description': 'Profound transformation.', 'strength': 0.86},
            'gateway': {'frequency': DAATH_FREQ*1.05, 'color': 'transparent', 'element': 'aether', 'keywords': ['portal', 'threshold', 'passage'], 'description': 'A gateway between realms.', 'strength': 0.87},
            'shadow_self': {'frequency': DAATH_FREQ*0.7, 'color': 'shadow', 'element': 'shadow', 'keywords': ['unconscious', 'hidden', 'integration'], 'description': 'Confronting the hidden aspects of self.', 'strength': 0.80},
        },
        'harmonic_signature_params': {'ratios': [1.0, 1.2, PHI, 1.8, 2.0, 2.618], 'falloff': 0.11}, # Complex, potentially dissonant harmonics
        'stability_modifier': 0.8, 'coherence_modifier': 0.8, 'energy_level': 0.82, 'chaos_order_bias': 0.7, 'resonance_multiplier': 0.98 # High chaos potential
    }
}

# --- END OF FILE sephiroth_data.py ---