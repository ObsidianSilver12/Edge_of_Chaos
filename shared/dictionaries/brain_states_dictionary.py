# Brain States and their characteristics to be used as modifiers for memory encoding and thought processing. interacts with signal patterns and energy dynamics.
# we may even use as modifiers to determine how the brain state impacts processing and output. example a hyper alert brain state would allow for higher volume
# training and more intense processing. system must monitor states to self regulate. too much training leading to errors or overload could indicate a state change is needed.
# things like processing intensity can impact energy value cost and things like emotional sensitivity could impact the emotional scoring, the brain wave frequency range could be used
# to model field dynamics when big state changes required. Within the system design fields will not be constant. the static field ia the only field constant. we will model field 
# dynamics against the initial field dynamics and will only model big changes. so big changes based on specific state changes will recalculate the field dynamics and that will be the new
# field object stored.

BRAIN_STATES = {
    # Waking States with Alertness Levels
    'hyper_alert': {
        'brain_state_id': 1,
        'brain_state_name': 'hyper_alert',
        'brain_state_description': 'Heightened awareness, danger response, or intense focus',
        'dominant_frequency_range': (20, 30),  # High Beta waves
        'processing_speed_modifier': 1.3,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.8,
        'default_processing_intensity': 0.9,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network alert triggering',
    },
    'focused': {
        'brain_state_id': 2,
        'brain_state_name': 'focused',
        'brain_state_description': 'Deliberate attention and concentration',
        'dominant_frequency_range': (15, 20),  # Beta waves
        'processing_speed_modifier': 1.2,
        'pattern_sensitivity': 0.8,
        'emotional_sensitivity': 0.6,
        'default_processing_intensity': 0.8,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network thinking triggering',
    },
    'relaxed_alert': {
        'brain_state_id': 3,
        'brain_state_name': 'relaxed_alert',
        'brain_state_description': 'Calm but attentive state',
        'dominant_frequency_range': (12, 15),  # Low Beta waves
        'processing_speed_modifier': 1.0,
        'pattern_sensitivity': 0.7,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.7,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on state change focus ended trigger',
    },
    'learning': {
        'brain_state_id': 4,
        'brain_state_name': 'learning',
        'brain_state_description': 'Active information acquisition and processing',
        'dominant_frequency_range': (10, 14),  # Beta-Alpha boundary
        'processing_speed_modifier': 1.1,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.6,
        'default_processing_intensity': 0.8,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on learning activated trigger which can be manual or triggered by model',
    },
    'autopilot': {
        'brain_state_id': 5,
        'brain_state_name': 'autopilot',
        'brain_state_description': 'Routine tasks with minimal conscious attention',
        'dominant_frequency_range': (8, 12),  # Alpha waves
        'processing_speed_modifier': 0.8,
        'pattern_sensitivity': 0.5,
        'emotional_sensitivity': 0.4,
        'default_processing_intensity': 0.5,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on autopilot trigger which activates after 30 minutes of inactivity after relaxed alert state activated',
    },
    'drowsy': {
        'brain_state_id': 6,
        'brain_state_name': 'drowsy',
        'brain_state_description': 'Reduced alertness, tired but awake',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.4,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.4,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on drowsy trigger which activates after 30 minutes of inactivity after autopilot state activated or after learning has been completed model gets drowsy',
    },
    
    # Sleep Cycle States
    'sleep_onset': {
        'brain_state_id': 7,
        'brain_state_name': 'sleep_onset',
        'brain_state_description': 'Transition from wakefulness to sleep',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.5,
        'pattern_sensitivity': 0.6,
        'emotional_sensitivity': 0.7,
        'default_processing_intensity': 0.4,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on sleep onset trigger activates 2 minutes after drowsy state activated',
    },
    'light_sleep_n1': {
        'brain_state_id': 8,
        'brain_state_name': 'light_sleep_n1',
        'brain_state_description': 'First stage of NREM sleep',
        'dominant_frequency_range': (4, 7),  # Theta waves
        'processing_speed_modifier': 0.4,
        'pattern_sensitivity': 0.5,
        'emotional_sensitivity': 0.6,
        'default_processing_intensity': 0.3,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on light sleep n1 trigger activates 2 minutes after sleep onset state activated',
    },
    'light_sleep_n2': {
        'brain_state_id': 9,
        'brain_state_name': 'light_sleep_n2',
        'brain_state_description': 'Second stage of NREM sleep with sleep spindles',
        'dominant_frequency_range': (12, 14),  # Sleep spindles
        'processing_speed_modifier': 0.3,
        'pattern_sensitivity': 0.4,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.3,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on light sleep n2 trigger activates 2 minutes after light sleep n1 state activated',
    },
    'deep_sleep_n3': {
        'brain_state_id': 10,
        'brain_state_name': 'deep_sleep_n3',
        'brain_state_description': 'Slow wave sleep, deep NREM',
        'dominant_frequency_range': (0.5, 4),  # Delta waves
        'processing_speed_modifier': 0.2,
        'pattern_sensitivity': 0.3,
        'emotional_sensitivity': 0.2,
        'default_processing_intensity': 0.2,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on deep sleep n3 trigger activates 2 minutes after light sleep n2 state activated',
    },
    'rem_light': {
        'brain_state_id': 11,
        'brain_state_name': 'rem_light',
        'brain_state_description': 'Early REM sleep with lower intensity',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.7,
        'emotional_sensitivity': 0.8,
        'default_processing_intensity': 0.5,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on rem light trigger activates 2 minutes after deep sleep n3 state activated',
    },
    'rem_intense': {
        'brain_state_id': 12,
        'brain_state_name': 'rem_intense',
        'brain_state_description': 'Deep REM sleep with vivid dreaming',
        'dominant_frequency_range': (20, 40),  # High frequency mixed
        'processing_speed_modifier': 0.7,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.9,
        'default_processing_intensity': 0.6,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on rem intense trigger activates 2 minutes after rem light state activated',
    },
    
    # Altered Consciousness States
    'meditation_light': {
        'brain_state_id': 13,
        'brain_state_name': 'meditation_light',
        'brain_state_description': 'Early stage meditation with relaxed awareness',
        'dominant_frequency_range': (8, 12),  # Alpha waves
        'processing_speed_modifier': 0.7,
        'pattern_sensitivity': 0.6,
        'emotional_sensitivity': 0.5,
        'default_processing_intensity': 0.4,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on meditation light trigger activated through a terminal command or model command',
    },
    'meditation_deep': {
        'brain_state_id': 14,
        'brain_state_name': 'meditation_deep',
        'brain_state_description': 'Deep meditation with heightened awareness',
        'dominant_frequency_range': (4, 7),  # Theta waves
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.8,
        'emotional_sensitivity': 0.7,
        'default_processing_intensity': 0.5,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on meditation deep trigger activated through a terminal command or model command',
    },
    'liminal_hypnagogic': {
        'brain_state_id': 15,
        'brain_state_name': 'liminal_hypnagogic',
        'brain_state_description': 'Transitional state between wakefulness and sleep',
        'dominant_frequency_range': (4, 8),  # Theta waves
        'processing_speed_modifier': 0.5,
        'pattern_sensitivity': 0.9,
        'emotional_sensitivity': 0.8,
        'default_processing_intensity': 0.6,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on liminal hypnagogic trigger activated through a terminal command or model command',
    },
    'liminal_hypnopompic': {
        'brain_state_id': 16,
        'brain_state_name': 'liminal_hypnopompic',
        'brain_state_description': 'Transitional state between sleep and wakefulness',
        'dominant_frequency_range': (4, 10),  # Theta-Alpha mix
        'processing_speed_modifier': 0.6,
        'pattern_sensitivity': 0.8,
        'emotional_sensitivity': 0.7,
        'default_processing_intensity': 0.5,
        'controller': 'mycelial_network',
        'controller_description': 'Controls the brain state through mycelial network on liminal hypnopompic trigger activates 2 minutes after liminal hypnagogic state activated',
    }
}