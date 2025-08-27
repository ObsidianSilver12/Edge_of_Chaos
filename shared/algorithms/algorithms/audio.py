import pandas as pd

# AUDITORY SENSE - COMPLETE ALGORITHM CATALOG
auditory_algorithms = {
    'Algorithm_Name': [],
    'Type': [],  # Model, Function, Methodology, Algorithm
    'Stage': [],  # SENSORY_RAW->PATTERNS, PATTERNS->FRAGMENTS, etc.
    'Category': [],
    'Description': [],
    'Mathematical_Basis': [],
    'Input_Data_Required': [],
    'Output_Data_Generated': [],
    'Computational_Complexity': [],
    'Biomimetic_Relevance': [],
    'Implementation_Notes': []
}

# =============================================================================
# STAGE 1: SENSORY_RAW -> PATTERNS
# =============================================================================

# All algorithm data as tuples
algorithms_data = [
    # SPECTRAL ANALYSIS ALGORITHMS
    ['Fast_Fourier_Transform', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Transforms time-domain audio signal into frequency domain for spectral analysis',
     'X(k) = Σ x(n) * e^(-j*2π*k*n/N) for k=0 to N-1',
     'audio_waveform, sampling_rate, window_function',
     'frequency_spectrum, dominant_frequencies, spectral_density',
     'O(n*log(n))', 5, 'Use scipy.fft or numpy.fft. Essential for all frequency analysis.'],
     
    ['Short_Time_Fourier_Transform', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Computes FFT over sliding windows to analyze time-varying frequency content',
     'STFT(m,ω) = Σ x(n) * w(n-m) * e^(-jωn)',
     'audio_waveform, window_size, hop_length, window_type',
     'spectrogram, time_frequency_representation, spectral_evolution',
     'O(n*log(w))', 5, 'Use librosa.stft(). Shows how frequency content changes over time.'],
     
    ['Mel_Frequency_Cepstral_Coefficients', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Extracts perceptually-relevant features using mel-scale frequency transformation',
     'MFCC = DCT(log(Mel_filterbank(|FFT(signal)|²)))',
     'audio_waveform, sampling_rate, n_mfcc, mel_filters',
     'mfcc_coefficients, mel_spectrogram, perceptual_features',
     'O(n*log(n))', 5, 'Use librosa.mfcc(). Highly biomimetic - models cochlea processing.'],
     
    ['Chromagram_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Maps frequency content to 12 pitch classes for harmonic analysis',
     'Chroma bins: map frequencies to 12 semitones using log frequency',
     'audio_waveform, sampling_rate, hop_length',
     'chromagram, pitch_class_profile, harmonic_content',
     'O(n*log(n))', 4, 'Use librosa.chroma_stft(). Essential for music analysis.'],
     
    ['Spectral_Centroid', 'Function', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Calculates the center of mass of the frequency spectrum',
     'Centroid = Σ(f * S(f)) / Σ S(f), where S(f) is spectral magnitude',
     'frequency_spectrum, spectral_magnitudes',
     'spectral_centroid, brightness_measure, timbral_feature',
     'O(n)', 4, 'Use librosa.spectral_centroid(). Indicates perceived brightness.'],
     
    ['Spectral_Rolloff', 'Function', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Finds frequency below which specified percentage of spectral energy is contained',
     'Rolloff frequency f_r where Σ(S(f≤f_r)) = p * Σ S(f), typically p=0.85',
     'frequency_spectrum, rolloff_percentage',
     'spectral_rolloff, frequency_cutoff, energy_distribution',
     'O(n)', 4, 'Use librosa.spectral_rolloff(). Shows frequency distribution shape.'],
     
    ['Spectral_Flux', 'Function', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Measures rate of change in spectral content between adjacent frames',
     'Flux = Σ(|S_t(f) - S_{t-1}(f)|) where S_t is spectrum at time t',
     'time_frequency_representation, frame_rate',
     'spectral_flux, spectral_change_rate, onset_detection',
     'O(n)', 4, 'Calculate frame-to-frame spectral differences. Good for onset detection.'],
     
    ['Harmonic_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Spectral_Analysis',
     'Identifies harmonic components and fundamental frequency in audio signal',
     'Autocorrelation or cepstral analysis to find periodic components',
     'audio_waveform, frequency_spectrum, harmonic_threshold',
     'fundamental_frequency, harmonic_frequencies, harmonic_content, pitch_estimation',
     'O(n*log(n))', 5, 'Use librosa.piptrack() or custom autocorrelation. Very biomimetic.'],

    # TEMPORAL ANALYSIS ALGORITHMS  
    ['Zero_Crossing_Rate', 'Function', 'SENSORY_RAW->PATTERNS', 'Temporal_Analysis',
     'Counts rate at which signal changes sign, indicating spectral characteristics',
     'ZCR = (1/2N) * Σ |sign(x(n)) - sign(x(n-1))|',
     'audio_waveform, frame_length',
     'zero_crossing_rate, spectral_characteristics, voiced_unvoiced_classification',
     'O(n)', 4, 'Simple but effective. High ZCR = noisy/fricative, Low ZCR = tonal.'],
     
    ['Root_Mean_Square_Energy', 'Function', 'SENSORY_RAW->PATTERNS', 'Temporal_Analysis',
     'Calculates RMS energy of audio signal over time windows',
     'RMS = √(Σ x(n)² / N) over sliding windows',
     'audio_waveform, window_size, hop_length',
     'rms_energy, energy_contour, loudness_estimation',
     'O(n)', 4, 'Use librosa.rms(). Correlates with perceived loudness.'],
     
    ['Onset_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Temporal_Analysis',
     'Detects beginning of musical notes or sound events using spectral change',
     'Peak picking on onset strength function derived from spectral flux',
     'audio_waveform, onset_threshold, pre_max, post_max',
     'onset_times, onset_strength, event_boundaries',
     'O(n*log(n))', 5, 'Use librosa.onset_detect(). Critical for rhythm and event analysis.'],
     
    ['Tempo_Estimation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Temporal_Analysis',
     'Estimates tempo (beats per minute) using onset detection and autocorrelation',
     'Autocorrelation of onset strength function to find periodic structure',
     'onset_times, onset_strength, tempo_range',
     'tempo_bpm, beat_tracking, rhythm_patterns',
     'O(n*log(n))', 4, 'Use librosa.tempo(). Combines onset detection with periodicity analysis.'],
     
    ['Beat_Tracking', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Temporal_Analysis',
     'Tracks beat locations using dynamic programming on onset strength',
     'Viterbi algorithm on onset strength with tempo constraints',
     'onset_strength, tempo_estimate, beat_threshold',
     'beat_times, beat_strength, rhythmic_structure',
     'O(n²)', 4, 'Use librosa.beat_track(). Complex but essential for music analysis.'],
     
    ['Silence_Detection', 'Function', 'SENSORY_RAW->PATTERNS', 'Temporal_Analysis',
     'Identifies silent or very quiet regions in audio based on energy thresholds',
     'Threshold-based classification: silence if RMS < threshold',
     'audio_waveform, silence_threshold, min_silence_duration',
     'silence_intervals, speech_activity, voice_activity_detection',
     'O(n)', 4, 'Essential for speech processing. Can use more sophisticated VAD methods.'],

    # PITCH AND FREQUENCY ANALYSIS
    ['Fundamental_Frequency_Estimation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Pitch_Analysis',
     'Estimates fundamental frequency (pitch) using autocorrelation or cepstral methods',
     'Autocorrelation: R(τ) = Σ x(n) * x(n+τ), find peak for F0 period',
     'audio_waveform, pitch_range, method_type',
     'fundamental_frequency, pitch_confidence, pitch_estimation',
     'O(n*log(n))', 5, 'Use librosa.yin() or pyin(). Core of pitch perception.'],
     
    ['Pitch_Contour_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Pitch_Analysis',
     'Tracks pitch changes over time for melody and intonation analysis',
     'Smooth pitch tracking with interpolation over voiced segments',
     'fundamental_frequency_sequence, smoothing_factor',
     'pitch_contour, melody_shape, intonation_patterns',
     'O(n)', 4, 'Connect F0 estimates across time. Important for speech and music.'],
     
    ['Pitch_Class_Histogram', 'Function', 'SENSORY_RAW->PATTERNS', 'Pitch_Analysis',
     'Creates histogram of pitch classes (C, C#, D, etc.) for tonal analysis',
     'Map all pitches to 12 semitone classes and count occurrences',
     'pitch_sequence, pitch_class_mapping',
     'pitch_class_distribution, tonal_center, key_estimation',
     'O(n)', 3, 'Useful for music key detection and tonal analysis.'],
     
    ['Vibrato_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Pitch_Analysis',
     'Analyzes periodic pitch modulation characteristic of vibrato',
     'Analyze pitch contour for periodic oscillations using FFT',
     'pitch_contour, vibrato_frequency_range',
     'vibrato_rate, vibrato_depth, pitch_modulation',
     'O(n*log(n))', 3, 'Apply FFT to pitch contour to find modulation frequency.'],

    # AMPLITUDE AND DYNAMICS ANALYSIS
    ['Dynamic_Range_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Amplitude_Analysis',
     'Analyzes the range between loudest and quietest parts of audio',
     'Dynamic Range = 20 * log10(max_rms / min_rms)',
     'rms_energy_sequence, percentile_analysis',
     'dynamic_range_db, loudness_variation, compression_detection',
     'O(n)', 3, 'Shows how much variation in loudness exists in the audio.'],
     
    ['Attack_Decay_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Amplitude_Analysis',
     'Analyzes attack and decay characteristics of sound onsets',
     'Exponential curve fitting to amplitude envelope after onset',
     'amplitude_envelope, onset_times, curve_fitting_method',
     'attack_time, decay_time, sustain_level, envelope_shape',
     'O(n)', 4, 'Important for instrument identification and synthesis.'],
     
    ['Loudness_Modeling', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Amplitude_Analysis',
     'Models perceived loudness using psychoacoustic principles',
     'Equal loudness contours and frequency masking effects',
     'frequency_spectrum, amplitude_spectrum, equal_loudness_curves',
     'perceived_loudness, loudness_sones, phon_scale',
     'O(n)', 5, 'Very biomimetic - models human auditory perception.'],

    # NOISE AND QUALITY ANALYSIS
    ['Signal_to_Noise_Ratio', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Calculates ratio between signal power and noise power',
     'SNR = 10 * log10(P_signal / P_noise) in dB',
     'audio_signal, noise_estimate, signal_estimate',
     'signal_to_noise_ratio, audio_quality, noise_level',
     'O(n)', 4, 'Essential quality metric. Requires signal/noise separation.'],
     
    ['Harmonic_to_Noise_Ratio', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Measures ratio of harmonic to noise components, important for voice analysis',
     'HNR = 10 * log10(P_harmonic / P_noise)',
     'audio_signal, harmonic_components, noise_components',
     'harmonic_to_noise_ratio, voice_quality, breathiness_measure',
     'O(n)', 4, 'Critical for voice quality assessment and pathology detection.'],
     
    ['Clipping_Detection', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Detects audio clipping distortion by identifying flat-topped waveforms',
     'Count samples at maximum amplitude for extended periods',
     'audio_waveform, clipping_threshold, minimum_duration',
     'clipping_detected, distortion_level, clipped_samples_percentage',
     'O(n)', 3, 'Simple threshold-based detection of amplitude clipping.'],

    # PSYCHOACOUSTIC ANALYSIS
    ['Critical_Band_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Psychoacoustic_Analysis',
     'Analyzes audio using critical bands that model human auditory frequency resolution',
     'Bark scale: 24 critical bands from 20Hz to 20kHz',
     'frequency_spectrum, bark_scale_mapping',
     'critical_bands, bark_spectrum, auditory_filter_response',
     'O(n)', 5, 'Highly biomimetic - models cochlear frequency analysis.'],
     
    ['Masking_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Psychoacoustic_Analysis',
     'Analyzes frequency masking effects where loud sounds mask quieter ones',
     'Psychoacoustic masking model with spreading function',
     'frequency_spectrum, masking_threshold, spreading_function',
     'masking_effects, audible_components, perceptual_relevance',
     'O(n²)', 4, 'Complex but important for perceptual audio analysis.'],
     
    ['Roughness_Calculation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Psychoacoustic_Analysis',
     'Calculates perceptual roughness caused by beating between close frequencies',
     'Model beating effects between spectral components',
     'frequency_spectrum, amplitude_spectrum, roughness_model',
     'roughness_measure, dissonance_level, beating_effects',
     'O(n²)', 3, 'Models perceptual roughness - important for music analysis.'],
     
    ['Sharpness_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Psychoacoustic_Analysis',
     'Measures perceptual sharpness related to high-frequency content',
     'Weight high frequencies more heavily in sharpness calculation',
     'frequency_spectrum, weighting_function',
     'sharpness_measure, high_frequency_emphasis, brightness_perception',
     'O(n)', 3, 'Models perception of high-frequency content prominence.'],

    # =============================================================================
    # STAGE 2: PATTERNS -> FRAGMENTS
    # =============================================================================
    
    ['Audio_Pattern_Segmentation', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Segments audio into coherent patterns based on spectral and temporal similarity',
     'Change point detection using spectral features and statistical methods',
     'spectral_features, temporal_features, segmentation_threshold',
     'audio_segments, segment_boundaries, pattern_coherence',
     'O(n²)', 4, 'Use change point detection algorithms like PELT or binary segmentation.'],
     
    ['Harmonic_Percussive_Separation', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Separates harmonic and percussive components using median filtering',
     'Median filtering in time (harmonic) and frequency (percussive) directions',
     'spectrogram, kernel_size, margin_factor',
     'harmonic_component, percussive_component, separated_sources',
     'O(n*log(n))', 4, 'Use librosa.hpss(). Useful for music analysis and source separation.'],
     
    ['Audio_Texture_Analysis', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Analyzes audio texture by combining spectral, temporal, and psychoacoustic features',
     'Statistical analysis of feature distributions over time windows',
     'mfcc_features, spectral_features, temporal_features, texture_window',
     'audio_texture_descriptor, texture_classification, temporal_texture',
     'O(n)', 4, 'Combine multiple feature types for comprehensive texture analysis.'],
     
    ['Cross_Modal_Audio_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Integrates audio patterns with other sensory modalities for unified understanding',
     'Correlation analysis and temporal alignment with other sensory inputs',
     'audio_patterns, visual_patterns, temporal_alignment, correlation_threshold',
     'cross_modal_correlations, synchronized_patterns, multi_modal_coherence',
     'O(n²)', 5, 'Essential for biomimetic multi-sensory processing.'],
     
    ['Audio_Event_Detection', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Detects and classifies discrete audio events within continuous streams',
     'Template matching or machine learning classification of audio segments',
     'audio_features, event_templates, classification_threshold',
     'detected_events, event_classifications, event_timestamps',
     'O(n*m)', 4, 'Use CNN or template matching for event classification.'],

    # =============================================================================
    # STAGE 3: FRAGMENTS -> NODES
    # =============================================================================
    
    ['Audio_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms abstract auditory concepts from integrated audio fragments',
     'Hierarchical clustering and concept abstraction from audio patterns',
     'audio_fragments, concept_hierarchy, abstraction_rules',
     'auditory_concepts, concept_relationships, concept_confidence',
     'O(n²)', 5, 'Create high-level auditory concepts - highly biomimetic.'],
     
    ['Sound_Source_Identification', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Identifies and categorizes sound sources from acoustic characteristics',
     'Multi-feature classification using spectral, temporal, and psychoacoustic cues',
     'audio_features, source_models, classification_confidence',
     'sound_source_categories, source_confidence, source_characteristics',
     'O(n)', 4, 'Combine multiple acoustic cues for robust source identification.'],
     
    ['Acoustic_Scene_Understanding', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Develops understanding of complete acoustic environment and context',
     'Integration of multiple sound sources and acoustic properties',
     'sound_sources, acoustic_properties, spatial_audio_cues',
     'acoustic_scene_description, environmental_context, scene_categories',
     'O(n²)', 4, 'Build comprehensive understanding of acoustic environment.'],
     
    ['Audio_Memory_Encoding', 'Function', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Encodes audio information for long-term memory storage and retrieval',
     'Compressed representation preserving perceptually important features',
     'auditory_concepts, importance_weights, encoding_strategy',
     'encoded_audio_memory, memory_keys, retrieval_cues',
     'O(n)', 4, 'Efficient encoding while preserving perceptually important information.'],

    # =============================================================================
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    # =============================================================================
    
    ['Audio_Embedding_Generation', 'Model', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Generates dense vector embeddings for audio content using neural networks',
     'Pre-trained audio embedding models like VGGish, OpenL3, or wav2vec',
     'audio_nodes, embedding_model, embedding_dimensions',
     'audio_embeddings, embedding_vectors, similarity_indices',
     'O(n)', 3, 'Use pre-trained models like VGGish or train custom embeddings.'],
     
    ['Audio_Similarity_Index', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates searchable index for audio similarity using embeddings and features',
     'Approximate nearest neighbor search using HNSW or LSH',
     'audio_embeddings, feature_vectors, similarity_metrics',
     'similarity_index, nearest_neighbor_structure, search_capability',
     'O(n*log(n))', 3, 'Use FAISS or similar for efficient similarity search.'],
     
    ['Audio_Tag_Generation', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Generates searchable tags and labels for audio content classification',
     'Automatic tagging based on audio analysis and classification results',
     'audio_analysis_results, classification_outputs, tag_vocabulary',
     'audio_tags, content_labels, searchable_descriptors',
     'O(n)', 3, 'Convert audio analysis results to searchable text tags.'],
     
    ['Audio_Fingerprinting', 'Algorithm', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates robust fingerprints for audio identification and matching',
     'Spectral peak constellation mapping or similar robust features',
     'audio_content, fingerprint_algorithm, hash_size',
     'audio_fingerprint, identification_hash, matching_capability',
     'O(n)', 4, 'Use algorithms like Shazam-style fingerprinting for identification.']
]

# Add all the data to the dictionary
for algo_data in algorithms_data:
    auditory_algorithms['Algorithm_Name'].append(algo_data[0])
    auditory_algorithms['Type'].append(algo_data[1])
    auditory_algorithms['Stage'].append(algo_data[2])
    auditory_algorithms['Category'].append(algo_data[3])
    auditory_algorithms['Description'].append(algo_data[4])
    auditory_algorithms['Mathematical_Basis'].append(algo_data[5])
    auditory_algorithms['Input_Data_Required'].append(algo_data[6])
    auditory_algorithms['Output_Data_Generated'].append(algo_data[7])
    auditory_algorithms['Computational_Complexity'].append(algo_data[8])
    auditory_algorithms['Biomimetic_Relevance'].append(algo_data[9])
    auditory_algorithms['Implementation_Notes'].append(algo_data[10])

# Create DataFrame
auditory_df = pd.DataFrame(auditory_algorithms)

# Display summary
print("AUDITORY SENSE - COMPLETE ALGORITHM CATALOG")
print("=" * 50)
print(f"Total Auditory Algorithms: {len(auditory_df)}")
print(f"Stage 1 (SENSORY_RAW->PATTERNS): {len(auditory_df[auditory_df['Stage'] == 'SENSORY_RAW->PATTERNS'])}")
print(f"Stage 2 (PATTERNS->FRAGMENTS): {len(auditory_df[auditory_df['Stage'] == 'PATTERNS->FRAGMENTS'])}")
print(f"Stage 3 (FRAGMENTS->NODES): {len(auditory_df[auditory_df['Stage'] == 'FRAGMENTS->NODES'])}")
print(f"Stage 4 (NODES->SEMANTIC_MAP): {len(auditory_df[auditory_df['Stage'] == 'NODES->SEMANTIC_MAP'])}")
print()

# Show category breakdown
print("ALGORITHM CATEGORIES:")
category_counts = auditory_df['Category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} algorithms")
print()

# Show type breakdown
print("ALGORITHM TYPES:")
type_counts = auditory_df['Type'].value_counts()
for algo_type, count in type_counts.items():
    print(f"  {algo_type}: {count} algorithms")
print()

# Show sample of the complete data
print("SAMPLE ALGORITHM DETAILS:")
print(auditory_df[['Algorithm_Name', 'Type', 'Stage', 'Category', 'Description']].head(10).to_string(index=False))

# Save to Excel if needed
# auditory_df.to_excel('auditory_algorithms_complete_catalog.xlsx', index=False)