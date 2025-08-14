# Cognitive states based on trigger events allow specific tools and algorithms to be utilised
# States include: learning, thinking, remembering, problem-solving, reasoning, reading, writing, conceptualising, 
# verbalising, singing, debating, deescalating, negotiating, planning, prioritising, constructing, deconstructing,
# meditating, soulsearching, resolving, transcending, sensing, feeling, sleeping, primordial
# Each state has unique properties that influence brain processing and is linked to specific brain states

COGNITIVE_STATES = {
    # Active Learning and Processing States
    'learning': {
        'cognitive_state_id': 1,
        'cognitive_state_name': 'learning',
        'cognitive_state_description': 'Active information acquisition and processing',
        'brain_wave_frequency': (10, 14),  # Beta-Alpha boundary
        'brain_wave_amplitude': (20, 40),  # Microvolts
        'brain_wave_pattern': {
            'type': 'synchronous_bursts',
            'description': 'Synchronized theta-alpha bursts (6-10Hz) with beta enhancement (12-14Hz) during encoding',
            'burst_frequency': (6, 10),  # Primary learning oscillation Hz
            'burst_duration': (500, 1000),  # milliseconds
            'enhancement_frequency': (12, 14),  # Cognitive enhancement Hz
            'coherence_pattern': 'hippocampal_cortical_synchrony'
        },
        'brain_sub_region_involvement': ['prefrontal_cortex', 'hippocampus', 'temporal_lobe'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls learning state through pattern recognition and memory consolidation',
        'tools': {
            'self_supervised': ['I-JEPA', 'V-JEPA', 'SimCLR', 'SwAV', 'BYOL', 'MoCo'],
            'neural_networks': ['Vision Transformer', 'Mamba', 'HyenaDNA', 'Neural ODE', 'SDE-Net'],
            'biomimetic': ['Spiking Neural Networks', 'Dendritic Computing', 'Mycelial Networks'],
            'multi_modal': ['CLIP', 'DALL-E', 'VideoCLIP', 'AudioCLIP', 'BioViL'],
            'frameworks': ['PyTorch', 'JAX', 'torchaudio', 'torchvision', 'torch-geometric'],
            'optimization': ['MUON', 'Adam', 'AdamW', 'Lion', 'Sophia'],
            'quality_assurance': ['Uncertainty Quantification', 'Deep Ensembles', 'MC Dropout'],
            'preprocessing': ['Mel-Spectrogram', 'MFCC', 'Data Augmentation', 'Contrastive Sampling']
        },
        'models': ['V-JEPA-1B', 'I-JEPA-ViT', 'Mamba-2.8B', 'HyenaDNA-1M', 'biomimetic_predictive_model'],
        'agents': ['self_supervised_agent', 'multi_modal_agent', 'quality_validation_agent'],
        'algorithms': {
            'self_supervised': ['Predictive Coding', 'Contrastive Learning', 'Masked Autoencoding', 'Joint Embedding'],
            'biomimetic': ['Spike-Timing-Dependent Plasticity', 'Dendritic Integration', 'Mycelial Logic'],
            'temporal_modeling': ['Mamba', 'State Space Models', 'Neural ODE', 'Continuous Time RNN'],
            'multi_modal': ['Cross-Modal Attention', 'Contrastive Alignment', 'Multi-Modal Fusion'],
            'meta_learning': ['MAML', 'Reptile', 'Few-Shot Learning', 'Continual Learning'],
            'quality_control': ['Forward-Backward Consistency', 'Expert Validation', 'Biological Plausibility']
        },
        'triggers': {
            'self_supervised': ['unlabeled_data_available', 'pattern_prediction_needed', 'representation_learning'],
            'multi_modal': ['cross_modal_alignment_needed', 'sensory_fusion_required', 'grounding_needed'],
            'quality_control': ['prediction_uncertainty_high', 'biological_implausibility', 'consistency_check_failed'],
            'adaptation': ['domain_shift_detected', 'new_modality_introduced', 'performance_degradation']
        },
        'brain_states': ['learning'],
        'associated_brain_state_ids': [4]
    },
    
    'thinking': {
        'cognitive_state_id': 2,
        'cognitive_state_name': 'thinking',
        'cognitive_state_description': 'Deliberate attention and concentration on problem solving',
        'brain_wave_frequency': (15, 20),  # Beta waves
        'brain_wave_amplitude': (15, 30),  # Microvolts
        'brain_wave_pattern': {
            'type': 'focused_coherence',
            'description': 'Sustained beta coherence (15-20Hz) across prefrontal networks with gamma bursts (30-40Hz)',
            'primary_frequency': (15, 20),  # Sustained attention Hz
            'gamma_bursts': (30, 40),  # Insight moments Hz
            'coherence_duration': (2000, 5000),  # milliseconds
            'coherence_pattern': 'frontal_parietal_network_sync'
        },
        'brain_sub_region_involvement': ['prefrontal_cortex', 'anterior_cingulate', 'parietal_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls thinking state through focused attention and cognitive processing',
        'tools': {
            'reasoning_engines': ['Chain-of-Thought', 'Tree-of-Thoughts', 'Graph-of-Thoughts', 'ReAct'],
            'biomimetic_reasoning': ['Neural ODEs', 'Liquid Neural Networks', 'Spiking Neural Networks'],
            'knowledge_graphs': ['Neo4j', 'PyKEEN', 'NetworkX', 'DGL', 'PyTorch Geometric'],
            'logical_reasoning': ['Prolog', 'ASP', 'SAT Solvers', 'SMT Solvers', 'Theorem Provers'],
            'neural_symbolic': ['Neural Module Networks', 'Graph Neural Networks', 'Differentiable Programming'],
            'memory_systems': ['Transformer Memory', 'External Memory Networks', 'Neural Turing Machines'],
            'hrm_systems': ['HRM-27M Implementation', 'Dual-Process Architecture', 'Single-Forward-Pass Reasoning'],
            'adaptive_computation': ['HBPO Framework', 'Budget-Constrained Hierarchies', 'Dynamic Reasoning Depth'],
            'frameworks': ['PySAT', 'OR-Tools', 'Z3', 'Choco', 'PyTorch', 'JAX', 'FlashAttention']
        },
        'models': ['GPT-4o-reasoning', 'Claude-3.5-Sonnet', 'Liquid-Net-L', 'Neural-ODE-Reasoning', 'HRM-27M', 'biomimetic_cortex'],
        'agents': ['reasoning_agent', 'logic_agent', 'creative_thinking_agent', 'meta_cognitive_agent'],
        'algorithms': {
            'neural_reasoning': ['Attention Mechanisms', 'Cross-Attention', 'Self-Attention', 'Multi-Head Attention'],
            'symbolic_reasoning': ['Forward Chaining', 'Backward Chaining', 'Resolution', 'Unification'],
            'biomimetic': ['Cortical Columns', 'Hierarchical Temporal Memory', 'Predictive Coding'],
            'graph_based': ['Graph Attention Networks', 'GraphSAGE', 'Graph Transformers', 'Message Passing'],
            'temporal': ['LSTM', 'GRU', 'Temporal Convolution', 'State Space Models', 'Mamba'],
            'meta_cognitive': ['Self-Monitoring', 'Strategy Selection', 'Cognitive Control', 'Working Memory'],
            'hrm_dual_process': ['High-Level Planning Module', 'Low-Level Execution Module', 'Recurrent Reasoning Cycles'],
            'adaptive_reasoning': ['HBPO Budget Optimization', 'Depth-Adaptive Computation', 'Problem-Complexity Matching'],
            'self_evolving': ['R-Zero Framework', 'Challenger-Solver Paradigm', 'Autonomous Data Generation'],
            'quality_assurance': ['Backward Reasoning', 'Forward Validation', 'Consistency Checking', 'Confidence Estimation']
        },
        'triggers': {
            'problem_detection': ['contradiction_found', 'incomplete_information', 'ambiguity_detected'],
            'reasoning_needed': ['complex_query', 'multi_step_problem', 'causal_reasoning_required'],
            'meta_cognitive': ['strategy_failure', 'confidence_low', 'need_verification'],
            'quality_control': ['inconsistency_detected', 'validation_required', 'backward_check_needed']
        },
        'brain_states': ['focused'],
        'associated_brain_state_ids': [2]
    },
    
    'remembering': {
        'cognitive_state_id': 3,
        'cognitive_state_name': 'remembering',
        'cognitive_state_description': 'Accessing and retrieving stored memories and experiences',
        'brain_wave_frequency': (8, 12),  # Alpha waves
        'brain_wave_amplitude': (25, 45),  # Microvolts
        'brain_wave_pattern': {
            'type': 'rhythmic_recall',
            'description': 'Alpha rhythm (8-12Hz) with theta reinstatement (4-8Hz) during memory retrieval',
            'alpha_rhythm': (8, 12),  # Background recall state Hz
            'theta_reinstatement': (4, 8),  # Memory reactivation Hz
            'replay_duration': (1000, 3000),  # milliseconds
            'coherence_pattern': 'hippocampal_neocortical_dialogue'
        },
        'brain_sub_region_involvement': ['hippocampus', 'temporal_lobe', 'frontal_cortex'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls memory retrieval through hippocampal-cortical networks',
        'tools': ['memory_retriever', 'context_matcher', 'temporal_sequencer'],
        'models': ['gpt-oss-20b-GGUF', 'biomimetic_memory_model', 'temporal_context_model'],
        'agents': ['memory_agent', 'recall_agent'],
        'algorithms': ['memory_search', 'context_matching', 'temporal_reconstruction'],
        'triggers': ['memory_query', 'context_cue', 'temporal_reference'],
        'brain_states': ['relaxed_alert', 'focused'],
        'associated_brain_state_ids': [3, 2]
    },
    
    # Problem Solving and Reasoning States
    'problem_solving': {
        'cognitive_state_id': 4,
        'cognitive_state_name': 'problem_solving',
        'cognitive_state_description': 'Complex analytical thinking and solution generation',
        'brain_wave_frequency': (20, 30),  # High Beta waves
        'brain_wave_amplitude': (10, 25),  # Microvolts
        'brain_wave_pattern': {
            'type': 'gamma_bursts',
            'description': 'High-frequency gamma bursts (30-100Hz) during insight with sustained beta (20-30Hz)',
            'beta_baseline': (20, 30),  # Analytical thinking Hz
            'gamma_bursts': (30, 100),  # Insight moments Hz
            'burst_duration': (100, 300),  # milliseconds
            'inter_burst_interval': (500, 2000),  # milliseconds
            'coherence_pattern': 'distributed_network_binding'
        },
        'brain_sub_region_involvement': ['prefrontal_cortex', 'anterior_cingulate', 'parietal_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls problem-solving through executive function networks',
        'tools': {
            'optimization': ['Scipy.optimize', 'CVXPY', 'PuLP', 'Gurobi', 'CPLEX'],
            'heuristic_search': ['Genetic Algorithm', 'Particle Swarm', 'Ant Colony', 'Simulated Annealing'],
            'machine_learning': ['XGBoost', 'Random Forest', 'SVM', 'Neural Networks'],
            'constraint_programming': ['OR-Tools', 'MiniZinc', 'Choco-Solver'],
            'graph_algorithms': ['NetworkX', 'Graph-Tool', 'DGL'],
            'numerical_methods': ['NumPy', 'SciPy', 'JAX', 'CuPy'],
            'visualization': ['Matplotlib', 'Plotly', 'Seaborn', 'NetworkX']
        },
        'models': ['gemm-3-12b-it-GGUF', 'Qwen3-30B-A3B-Instruct-2507', 'biomimetic_reasoning_model'],
        'agents': ['problem_solver_agent', 'strategy_agent'],
        'algorithms': {
            'search': ['A*', 'IDA*', 'Beam Search', 'Monte Carlo Tree Search', 'Alpha-Beta Pruning'],
            'optimization': ['Gradient Descent', 'Genetic Algorithm', 'Particle Swarm', 'Differential Evolution'],
            'constraint_satisfaction': ['Backtracking', 'Forward Checking', 'Arc Consistency', 'Local Search'],
            'machine_learning': ['Decision Trees', 'Random Forest', 'Gradient Boosting', 'Neural Networks'],
            'heuristic': ['Greedy', 'Hill Climbing', 'Simulated Annealing', 'Tabu Search'],
            'graph': ['Dijkstra', 'Floyd-Warshall', 'Minimum Spanning Tree', 'Maximum Flow']
        },
        'triggers': {
            'complexity': ['exponential_search_space', 'NP_complete_problem', 'multi_objective_optimization'],
            'constraints': ['resource_limitation', 'time_constraint', 'feasibility_issue'],
            'performance': ['suboptimal_solution', 'local_minimum_trapped', 'convergence_failure'],
            'domain': ['new_problem_type', 'domain_specific_constraints', 'expert_knowledge_needed']
        },
        'brain_states': ['hyper_alert', 'focused'],
        'associated_brain_state_ids': [1, 2]
    },
    
    'reasoning': {
        'cognitive_state_id': 5,
        'cognitive_state_name': 'reasoning',
        'cognitive_state_description': 'Logical analysis and inference making',
        'brain_wave_frequency': (15, 25),  # Beta waves
        'brain_wave_amplitude': (15, 30),  # Microvolts
        'brain_wave_pattern': 'sequential_processing',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'temporal_lobe', 'parietal_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls reasoning through logical processing networks',
        'tools': ['inference_engine', 'logic_validator', 'premise_analyzer'],
        'models': ['reasoning_model', 'inference_model'],
        'agents': ['reasoning_agent', 'logic_agent'],
        'algorithms': ['logical_inference', 'premise_validation', 'conclusion_derivation'],
        'triggers': ['logical_query', 'inference_needed', 'validation_required'],
        'brain_states': ['focused', 'learning'],
        'associated_brain_state_ids': [2, 4]
    },
    
    # Communication and Expression States
    'reading': {
        'cognitive_state_id': 6,
        'cognitive_state_name': 'reading',
        'cognitive_state_description': 'Text processing and comprehension',
        'brain_wave_frequency': (12, 18),  # Beta-Alpha mix
        'brain_wave_amplitude': (20, 35),  # Microvolts
        'brain_wave_pattern': {
            'type': 'scanning_rhythm',
            'description': 'Rhythmic 2-4Hz saccadic eye movements overlaid on 12-18Hz cognitive processing waves',
            'saccade_frequency': (2, 4),  # Eye movement rhythm Hz
            'fixation_duration': (200, 400),  # milliseconds
            'cognitive_frequency': (12, 18),  # Background processing Hz
            'coherence_pattern': 'visual_linguistic_synchrony'
        },
        'brain_sub_region_involvement': ['visual_cortex', 'language_areas', 'temporal_lobe'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls reading through visual-linguistic processing',
        'tools': {
            'nlp_libraries': ['spaCy', 'NLTK', 'transformers', 'gensim', 'textblob'],
            'tokenizers': ['BPE', 'WordPiece', 'SentencePiece', 'tiktoken'],
            'embeddings': ['Word2Vec', 'GloVe', 'FastText', 'BERT embeddings', 'Sentence Transformers'],
            'parsers': ['Dependency Parser', 'Constituency Parser', 'NER', 'POS Tagger'],
            'ocr_tools': ['Tesseract', 'EasyOCR', 'PaddleOCR', 'TrOCR'],
            'preprocessing': ['Text Cleaning', 'Normalization', 'Stemming', 'Lemmatization'],
            'frameworks': ['Hugging Face', 'AllenNLP', 'Flair', 'Stanza']
        },
        'models': ['gemma-3n-E4B-it-GGUF', 'gpt-oss-20b-GGUF'],
        'agents': ['reading_agent', 'comprehension_agent'],
        'algorithms': {
            'text_processing': ['Tokenization', 'Stemming', 'Lemmatization', 'N-gram Analysis'],
            'language_models': ['BERT', 'RoBERTa', 'DistilBERT', 'ELECTRA', 'T5'],
            'sequence_models': ['BiLSTM', 'GRU', 'Transformer', 'CNN'],
            'classification': ['Naive Bayes', 'SVM', 'Logistic Regression', 'Neural Networks'],
            'parsing': ['CKY', 'Earley', 'LR', 'Shift-Reduce'],
            'attention': ['Self-Attention', 'Cross-Attention', 'Sparse Attention', 'Local Attention']
        },
        'triggers': {
            'input_type': ['text_document', 'pdf_file', 'image_with_text', 'handwritten_text'],
            'complexity': ['technical_document', 'multilingual_text', 'domain_specific_terminology'],
            'quality': ['low_resolution_text', 'noisy_ocr', 'formatting_issues'],
            'comprehension': ['ambiguous_meaning', 'context_dependent', 'inference_required']
        },
        'brain_states': ['focused', 'relaxed_alert'],
        'associated_brain_state_ids': [2, 3]
    },
    
    'writing': {
        'cognitive_state_id': 7,
        'cognitive_state_name': 'writing',
        'cognitive_state_description': 'Text generation and composition',
        'brain_wave_frequency': (14, 22),  # Beta waves
        'brain_wave_amplitude': (18, 32),  # Microvolts
        'brain_wave_pattern': {
            'type': 'creative_flow',
            'description': 'Alpha-beta transition (10-18Hz) with creative bursts and motor planning (22-30Hz)',
            'alpha_creativity': (10, 14),  # Creative ideation Hz
            'beta_execution': (14, 22),  # Language execution Hz
            'motor_planning': (22, 30),  # Writing motor preparation Hz
            'flow_duration': (3000, 10000),  # milliseconds
            'coherence_pattern': 'language_motor_integration'
        },
        'brain_sub_region_involvement': ['language_areas', 'motor_cortex', 'prefrontal_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls writing through linguistic and motor coordination',
        'tools': ['text_generator', 'grammar_checker', 'style_optimizer'],
        'models': ['gpt-oss-20b-GGUF', 'gemma-3n-E4B-it-GGUF', 'biomimetic_language_model'],
        'agents': ['writing_agent', 'editor_agent', 'language_assistant_agent'],
        'algorithms': ['text_generation', 'grammar_validation', 'style_optimization'],
        'triggers': ['writing_request', 'composition_needed', 'text_output_required'],
        'brain_states': ['focused', 'learning'],
        'associated_brain_state_ids': [2, 4]
    },
    
    'conceptualising': {
        'cognitive_state_id': 8,
        'cognitive_state_name': 'conceptualising',
        'cognitive_state_description': 'Abstract thinking and concept formation',
        'brain_wave_frequency': (8, 15),  # Alpha-Beta boundary
        'brain_wave_amplitude': (25, 40),  # Microvolts
        'brain_wave_pattern': {
            'type': 'creative_synthesis',
            'description': 'Alpha-theta crossover (6-12Hz) with beta integration (12-18Hz) during concept formation',
            'theta_creativity': (6, 8),  # Deep creative access Hz
            'alpha_ideation': (8, 12),  # Conceptual thinking Hz
            'beta_integration': (12, 18),  # Concept integration Hz
            'synthesis_duration': (2000, 8000),  # milliseconds
            'coherence_pattern': 'default_mode_executive_coupling'
        },
        'brain_sub_region_involvement': ['prefrontal_cortex', 'temporal_lobe', 'parietal_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls conceptualisation through abstract thinking networks',
        'tools': ['concept_mapper', 'abstraction_engine', 'pattern_synthesizer'],
        'models': ['Qwen3-30B-A3B-Instruct-2507', 'biomimetic_concept_model', 'abstract_reasoning_model'],
        'agents': ['concept_agent', 'abstraction_agent', 'synthesis_agent'],
        'algorithms': ['concept_formation', 'abstraction', 'pattern_synthesis'],
        'triggers': ['abstract_thinking_needed', 'concept_formation_required', 'pattern_detected'],
        'brain_states': ['focused', 'relaxed_alert'],
        'associated_brain_state_ids': [2, 3]
    },
    
    'verbalising': {
        'cognitive_state_id': 9,
        'cognitive_state_name': 'verbalising',
        'cognitive_state_description': 'Speech production and verbal expression',
        'brain_wave_frequency': (16, 24),  # Beta waves
        'brain_wave_amplitude': (15, 28),  # Microvolts
        'brain_wave_pattern': 'motor_linguistic',
        'brain_sub_region_involvement': ['broca_area', 'motor_cortex', 'auditory_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls verbalisation through speech motor networks',
        'tools': ['speech_synthesizer', 'phonetic_processor', 'prosody_controller'],
        'models': ['speech_model', 'phonetic_model'],
        'agents': ['speech_agent', 'verbal_agent'],
        'algorithms': ['speech_synthesis', 'phonetic_processing', 'prosody_generation'],
        'triggers': ['speech_request', 'verbal_output_needed', 'vocalization_required'],
        'brain_states': ['focused', 'relaxed_alert'],
        'associated_brain_state_ids': [2, 3]
    },
    
    # Creative and Expressive States
    'singing': {
        'cognitive_state_id': 10,
        'cognitive_state_name': 'singing',
        'cognitive_state_description': 'Musical expression and vocal performance',
        'brain_wave_frequency': (8, 20),  # Alpha-Beta mix
        'brain_wave_amplitude': (20, 35),  # Microvolts
        'brain_wave_pattern': {
            'type': 'rhythmic_melodic',
            'description': 'Alpha rhythm (8-12Hz) with beta motor coordination (16-24Hz) and emotional modulation',
            'alpha_musical': (8, 12),  # Musical processing Hz
            'beta_motor': (16, 24),  # Vocal motor control Hz
            'emotional_modulation': (4, 8),  # Theta emotional expression Hz
            'rhythmic_entrainment': 'tempo_locked',  # Locks to musical tempo
            'coherence_pattern': 'auditory_motor_emotional_sync'
        },
        'brain_sub_region_involvement': ['auditory_cortex', 'motor_cortex', 'emotional_centers'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls singing through musical-motor coordination',
        'tools': ['melody_generator', 'rhythm_controller', 'vocal_modulator'],
        'models': ['biomimetic_audio_model', 'musical_generation_model', 'vocal_synthesis_model'],
        'agents': ['musical_agent', 'vocal_agent', 'rhythm_agent'],
        'algorithms': ['melody_generation', 'rhythm_processing', 'vocal_modulation'],
        'triggers': ['musical_expression_needed', 'vocal_performance_request', 'emotional_expression'],
        'brain_states': ['relaxed_alert', 'focused'],
        'associated_brain_state_ids': [3, 2]
    },
    
    # Social and Interpersonal States
    'debating': {
        'cognitive_state_id': 11,
        'cognitive_state_name': 'debating',
        'cognitive_state_description': 'Argumentative discourse and position defense',
        'brain_wave_frequency': (18, 28),  # High Beta waves
        'brain_wave_amplitude': (12, 25),  # Microvolts
        'brain_wave_pattern': 'competitive_alertness',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'language_areas', 'emotional_centers'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls debating through competitive reasoning networks',
        'tools': ['argument_analyzer', 'counter_argument_generator', 'rhetoric_optimizer'],
        'models': ['debate_model', 'argumentation_model'],
        'agents': ['debate_agent', 'rhetoric_agent'],
        'algorithms': ['argument_construction', 'counter_argument_generation', 'rhetorical_analysis'],
        'triggers': ['debate_initiated', 'argument_challenge', 'position_defense_needed'],
        'brain_states': ['hyper_alert', 'focused'],
        'associated_brain_state_ids': [1, 2]
    },
    
    'deescalating': {
        'cognitive_state_id': 12,
        'cognitive_state_name': 'deescalating',
        'cognitive_state_description': 'Conflict resolution and tension reduction',
        'brain_wave_frequency': (8, 14),  # Alpha-low Beta
        'brain_wave_amplitude': (25, 40),  # Microvolts
        'brain_wave_pattern': 'calming_coherence',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'emotional_centers', 'empathy_networks'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls deescalation through empathy and emotional regulation',
        'tools': ['empathy_analyzer', 'emotion_regulator', 'conflict_resolver'],
        'models': ['deescalation_model', 'empathy_model'],
        'agents': ['mediator_agent', 'empathy_agent'],
        'algorithms': ['emotion_regulation', 'empathy_modeling', 'conflict_resolution'],
        'triggers': ['conflict_detected', 'tension_rising', 'mediation_needed'],
        'brain_states': ['relaxed_alert', 'focused'],
        'associated_brain_state_ids': [3, 2]
    },
    
    'negotiating': {
        'cognitive_state_id': 13,
        'cognitive_state_name': 'negotiating',
        'cognitive_state_description': 'Strategic bargaining and agreement seeking',
        'brain_wave_frequency': (15, 22),  # Beta waves
        'brain_wave_amplitude': (18, 30),  # Microvolts
        'brain_wave_pattern': 'strategic_planning',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'social_cognition_areas', 'reward_centers'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls negotiation through strategic social cognition',
        'tools': ['strategy_planner', 'value_assessor', 'compromise_finder'],
        'models': ['negotiation_model', 'strategy_model'],
        'agents': ['negotiator_agent', 'strategy_agent'],
        'algorithms': ['strategic_planning', 'value_assessment', 'compromise_optimization'],
        'triggers': ['negotiation_initiated', 'agreement_needed', 'conflict_resolution_required'],
        'brain_states': ['focused', 'relaxed_alert'],
        'associated_brain_state_ids': [2, 3]
    },
    
    # Planning and Organization States
    'planning': {
        'cognitive_state_id': 14,
        'cognitive_state_name': 'planning',
        'cognitive_state_description': 'Future-oriented goal setting and strategy development',
        'brain_wave_frequency': (12, 18),  # Beta waves
        'brain_wave_amplitude': (20, 35),  # Microvolts
        'brain_wave_pattern': 'sequential_organization',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'parietal_cortex', 'temporal_lobe'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls planning through executive function networks',
        'tools': ['goal_setter', 'timeline_manager', 'resource_allocator'],
        'models': ['planning_model', 'goal_achievement_model'],
        'agents': ['planning_agent', 'goal_agent'],
        'algorithms': ['goal_decomposition', 'timeline_optimization', 'resource_allocation'],
        'triggers': ['goal_set', 'planning_required', 'future_preparation_needed'],
        'brain_states': ['focused', 'learning'],
        'associated_brain_state_ids': [2, 4]
    },
    
    'prioritising': {
        'cognitive_state_id': 15,
        'cognitive_state_name': 'prioritising',
        'cognitive_state_description': 'Importance ranking and decision making',
        'brain_wave_frequency': (14, 20),  # Beta waves
        'brain_wave_amplitude': (16, 28),  # Microvolts
        'brain_wave_pattern': 'evaluative_processing',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'anterior_cingulate', 'reward_centers'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls prioritisation through value assessment networks',
        'tools': ['value_assessor', 'importance_ranker', 'decision_matrix'],
        'models': ['priority_model', 'decision_model'],
        'agents': ['priority_agent', 'decision_agent'],
        'algorithms': ['value_assessment', 'importance_ranking', 'decision_optimization'],
        'triggers': ['choice_required', 'resource_constraint', 'priority_conflict'],
        'brain_states': ['focused', 'relaxed_alert'],
        'associated_brain_state_ids': [2, 3]
    },
    
    # Construction and Creation States
    'constructing': {
        'cognitive_state_id': 16,
        'cognitive_state_name': 'constructing',
        'cognitive_state_description': 'Building and assembling complex structures or ideas',
        'brain_wave_frequency': (10, 16),  # Alpha-Beta boundary
        'brain_wave_amplitude': (22, 38),  # Microvolts
        'brain_wave_pattern': 'creative_building',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'parietal_cortex', 'motor_areas'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls construction through spatial-motor coordination',
        'tools': ['structure_builder', 'component_assembler', 'design_optimizer'],
        'models': ['construction_model', 'design_model'],
        'agents': ['builder_agent', 'design_agent'],
        'algorithms': ['structure_assembly', 'component_integration', 'design_optimization'],
        'triggers': ['construction_needed', 'assembly_required', 'building_initiated'],
        'brain_states': ['focused', 'learning'],
        'associated_brain_state_ids': [2, 4]
    },
    
    'deconstructing': {
        'cognitive_state_id': 17,
        'cognitive_state_name': 'deconstructing',
        'cognitive_state_description': 'Breaking down complex structures into components',
        'brain_wave_frequency': (16, 24),  # Beta waves
        'brain_wave_amplitude': (14, 26),  # Microvolts
        'brain_wave_pattern': 'analytical_breakdown',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'parietal_cortex', 'temporal_lobe'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls deconstruction through analytical processing',
        'tools': ['structure_analyzer', 'component_extractor', 'relationship_mapper'],
        'models': ['deconstruction_model', 'analysis_model'],
        'agents': ['analyzer_agent', 'breakdown_agent'],
        'algorithms': ['structure_analysis', 'component_extraction', 'relationship_mapping'],
        'triggers': ['analysis_required', 'breakdown_needed', 'component_identification'],
        'brain_states': ['focused', 'hyper_alert'],
        'associated_brain_state_ids': [2, 1]
    },
    
    # Contemplative and Spiritual States
    'meditating': {
        'cognitive_state_id': 18,
        'cognitive_state_name': 'meditating',
        'cognitive_state_description': 'Contemplative awareness and mindfulness practice',
        'brain_wave_frequency': (4, 8),  # Theta waves
        'brain_wave_amplitude': (30, 50),  # Microvolts
        'brain_wave_pattern': 'meditative_coherence',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'anterior_cingulate', 'default_mode_network'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls meditation through mindfulness networks',
        'tools': ['awareness_monitor', 'mindfulness_tracker', 'attention_stabilizer'],
        'models': ['meditation_model', 'mindfulness_model'],
        'agents': ['meditation_agent', 'mindfulness_agent'],
        'algorithms': ['attention_regulation', 'awareness_monitoring', 'mindfulness_cultivation'],
        'triggers': ['meditation_initiated', 'mindfulness_practice', 'contemplation_needed'],
        'brain_states': ['meditation_light', 'meditation_deep'],
        'associated_brain_state_ids': [13, 14]
    },
    
    'soulsearching': {
        'cognitive_state_id': 19,
        'cognitive_state_name': 'soulsearching',
        'cognitive_state_description': 'Deep introspection and self-discovery',
        'brain_wave_frequency': (4, 10),  # Theta-Alpha mix
        'brain_wave_amplitude': (35, 55),  # Microvolts
        'brain_wave_pattern': 'introspective_depth',
        'brain_sub_region_involvement': ['default_mode_network', 'limbic_system', 'prefrontal_cortex'],
        'grid': 'brain_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls soul searching through deep introspective networks',
        'tools': ['self_analyzer', 'value_explorer', 'meaning_finder'],
        'models': ['introspection_model', 'self_discovery_model'],
        'agents': ['introspection_agent', 'self_discovery_agent'],
        'algorithms': ['self_analysis', 'value_exploration', 'meaning_construction'],
        'triggers': ['identity_crisis', 'life_transition', 'meaning_seeking'],
        'brain_states': ['meditation_deep', 'liminal_hypnagogic'],
        'associated_brain_state_ids': [14, 15]
    },
    
    'resolving': {
        'cognitive_state_id': 20,
        'cognitive_state_name': 'resolving',
        'cognitive_state_description': 'Conflict resolution and problem settlement',
        'brain_wave_frequency': (8, 14),  # Alpha-low Beta
        'brain_wave_amplitude': (25, 40),  # Microvolts
        'brain_wave_pattern': 'resolution_seeking',
        'brain_sub_region_involvement': ['prefrontal_cortex', 'anterior_cingulate', 'emotional_centers'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls resolution through conflict processing networks',
        'tools': ['conflict_analyzer', 'solution_finder', 'harmony_creator'],
        'models': ['resolution_model', 'conflict_model'],
        'agents': ['resolution_agent', 'harmony_agent'],
        'algorithms': ['conflict_analysis', 'solution_generation', 'harmony_optimization'],
        'triggers': ['conflict_identified', 'resolution_needed', 'harmony_sought'],
        'brain_states': ['focused', 'relaxed_alert'],
        'associated_brain_state_ids': [2, 3]
    },
    
    'transcending': {
        'cognitive_state_id': 21,
        'cognitive_state_name': 'transcending',
        'cognitive_state_description': 'Moving beyond limitations and achieving higher understanding',
        'brain_wave_frequency': (4, 7),  # Theta waves
        'brain_wave_amplitude': (40, 60),  # Microvolts
        'brain_wave_pattern': 'transcendent_unity',
        'brain_sub_region_involvement': ['default_mode_network', 'temporal_lobe', 'parietal_cortex'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls transcendence through expanded awareness networks',
        'tools': ['limitation_dissolver', 'perspective_expander', 'unity_perceiver'],
        'models': ['transcendence_model', 'unity_model'],
        'agents': ['transcendence_agent', 'unity_agent'],
        'algorithms': ['limitation_transcendence', 'perspective_expansion', 'unity_perception'],
        'triggers': ['limitation_encountered', 'transcendence_sought', 'unity_experienced'],
        'brain_states': ['meditation_deep', 'liminal_hypnagogic'],
        'associated_brain_state_ids': [14, 15]
    },
    
    # Sensory and Perceptual States
    'sensing': {
        'cognitive_state_id': 22,
        'cognitive_state_name': 'sensing',
        'cognitive_state_description': 'Active sensory perception and environmental awareness',
        'brain_wave_frequency': (12, 25),  # Beta waves
        'brain_wave_amplitude': (15, 30),  # Microvolts
        'brain_wave_pattern': {
            'type': 'sensory_alertness',
            'description': 'Multi-frequency sensory processing with alpha gating (8-12Hz) and beta alertness (12-25Hz)',
            'alpha_gating': (8, 12),  # Sensory gating Hz
            'beta_alertness': (12, 25),  # Active sensing Hz
            'gamma_binding': (30, 80),  # Feature binding Hz
            'processing_cycles': (50, 200),  # milliseconds per cycle
            'coherence_pattern': 'sensory_integration_network'
        },
        'brain_sub_region_involvement': ['sensory_cortices', 'thalamus', 'parietal_cortex'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls sensing through sensory processing networks',
        'tools': {
            'computer_vision': ['V-JEPA', 'DINO', 'DINOv2', 'MAE', 'SimMIM'],
            'biomimetic_vision': ['Gabor Filter Banks', 'Retinal Preprocessing', 'Saccadic Processing'],
            'real_time': ['OpenCV', 'MediaPipe', 'TensorRT', 'ONNX Runtime', 'CoreML'],
            'audio_processing': ['Mamba-Audio', 'Wav2Vec', 'HuBERT', 'WavLM', 'Whisper'],
            'multi_modal': ['CLIP', 'AudioCLIP', 'VideoCLIP', 'ImageBind', 'LanguageBind'],
            'neuromorphic': ['DVS Camera Processing', 'Event-Based Vision', 'Spiking Vision'],
            'frameworks': ['torchvision', 'torchaudio', 'mmdetection', 'detectron2', 'kornia']
        },
        'models': ['V-JEPA-ViT-H', 'DINOv2-ViT-L', 'Mamba-Audio-L', 'CLIP-ViT-L', 'biomimetic_sensor_fusion'],
        'agents': ['visual_processing_agent', 'audio_processing_agent', 'sensor_fusion_agent', 'attention_control_agent'],
        'algorithms': {
            'visual_processing': ['Self-Supervised ViT', 'Masked Image Modeling', 'Contrastive Learning', 'Object-Centric Learning'],
            'audio_processing': ['Mel-Spectrogram Analysis', 'Temporal Convolution', 'Audio Transformer', 'Self-Supervised Audio'],
            'biomimetic': ['Retinal Ganglion Cell Simulation', 'V1 Simple/Complex Cells', 'Attention Mechanisms'],
            'temporal': ['Optical Flow', 'Motion Estimation', 'Temporal Coherence', 'Predictive Coding'],
            'fusion': ['Early Fusion', 'Late Fusion', 'Attention Fusion', 'Cross-Modal Attention'],
            'neuromorphic': ['Event-Based Processing', 'Spike-Based Encoding', 'Temporal Difference Coding']
        },
        'triggers': {
            'visual_input': ['frame_received', 'motion_detected', 'object_appeared', 'attention_shift'],
            'audio_input': ['sound_onset', 'frequency_change', 'pattern_recognized', 'silence_broken'],
            'cross_modal': ['audio_visual_sync', 'multi_modal_conflict', 'sensory_binding_needed'],
            'quality_issues': ['low_light_detected', 'motion_blur', 'audio_noise', 'sensor_malfunction']
        },
        'brain_states': ['hyper_alert', 'focused'],
        'associated_brain_state_ids': [1, 2]
    },
    
    'feeling': {
        'cognitive_state_id': 23,
        'cognitive_state_name': 'feeling',
        'cognitive_state_description': 'Emotional processing and affective experience',
        'brain_wave_frequency': (6, 12),  # Theta-Alpha waves
        'brain_wave_amplitude': (30, 45),  # Microvolts
        'brain_wave_pattern': {
            'type': 'emotional_resonance',
            'description': 'Theta-alpha emotional processing (6-12Hz) with amygdala-prefrontal coherence',
            'emotional_theta': (4, 8),  # Deep emotional processing Hz
            'cognitive_alpha': (8, 12),  # Emotional awareness Hz
            'resonance_duration': (2000, 8000),  # milliseconds
            'coherence_pattern': 'limbic_cortical_integration'
        },
        'brain_sub_region_involvement': ['limbic_system', 'amygdala', 'prefrontal_cortex', 'anterior_cingulate'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls feeling through emotional processing and regulation networks',
        'tools': {
            'emotion_ai': ['FER-2013', 'EmotiW', 'RAVDESS', 'IEMOCAP', 'OpenFace'],
            'biomimetic_emotion': ['Amygdala Simulation', 'Limbic System Models', 'Neurochemical Modeling'],
            'multimodal_emotion': ['Audio-Visual Emotion', 'Text-Emotion', 'Physiological Signals'],
            'real_time': ['OpenCV Emotion', 'MediaPipe Face', 'Speech Emotion Recognition'],
            'neuromorphic': ['Spiking Emotional Networks', 'Event-Based Emotion', 'Neuromorphic Affective Computing'],
            'frameworks': ['PyTorch Emotion', 'TensorFlow Emotion', 'Hugging Face Emotion', 'scikit-emotion']
        },
        'models': ['BERT-Emotion', 'GPT-Emotion', 'Wav2Vec-Emotion', 'ViT-Emotion', 'biomimetic_limbic_system'],
        'agents': ['emotion_recognition_agent', 'affect_regulation_agent', 'empathy_modeling_agent', 'emotional_memory_agent'],
        'algorithms': {
            'emotion_recognition': ['CNN-LSTM', 'Transformer-Emotion', 'Multi-Modal Fusion', 'Attention-Based Recognition'],
            'biomimetic_processing': ['Amygdala-Prefrontal Loops', 'Dopamine-Serotonin Modeling', 'Limbic Resonance'],
            'regulation': ['Cognitive Reappraisal', 'Emotion Regulation Networks', 'Prefrontal Control'],
            'temporal_dynamics': ['Emotion Trajectories', 'Affective State Transitions', 'Emotional Memory Integration'],
            'quality_assurance': ['Emotion Validation', 'Cross-Modal Consistency', 'Biological Plausibility Check']
        },
        'triggers': {
            'emotional_input': ['facial_expression_detected', 'voice_emotion_detected', 'text_sentiment_analyzed'],
            'internal_states': ['memory_emotional_trigger', 'physiological_arousal', 'cognitive_appraisal'],
            'social_cues': ['empathy_activation', 'social_mirroring', 'emotional_contagion'],
            'regulation_needed': ['emotional_intensity_high', 'conflicting_emotions', 'maladaptive_response']
        },
        'brain_states': ['relaxed_alert', 'focused'],
        'associated_brain_state_ids': [3, 2]
    },
    
    # Sleep and Recovery States
    'sleeping': {
        'cognitive_state_id': 24,
        'cognitive_state_name': 'sleeping',
        'cognitive_state_description': 'Rest, recovery, and memory consolidation during sleep cycles',
        'brain_wave_frequency': (0.5, 40),  # Full spectrum depending on sleep stage
        'brain_wave_amplitude': (20, 100),  # Microvolts - varies by stage
        'brain_wave_pattern': {
            'type': 'cyclical_sleep_stages',
            'description': 'Cyclical progression through NREM and REM stages with distinct wave patterns',
            'stage_patterns': {
                'sleep_onset': {'frequency': (4, 8), 'amplitude': (30, 50), 'duration': 120},  # seconds
                'light_sleep_n1': {'frequency': (4, 7), 'amplitude': (25, 45), 'duration': 300},
                'light_sleep_n2': {'frequency': (12, 14), 'amplitude': (20, 40), 'duration': 600},  # sleep spindles
                'deep_sleep_n3': {'frequency': (0.5, 4), 'amplitude': (50, 100), 'duration': 1200},  # delta waves
                'rem_light': {'frequency': (4, 8), 'amplitude': (15, 30), 'duration': 600},
                'rem_intense': {'frequency': (20, 40), 'amplitude': (10, 25), 'duration': 900}
            },
            'cycle_duration': (90, 120),  # minutes per complete cycle
            'coherence_pattern': 'thalamocortical_oscillations'
        },
        'brain_sub_region_involvement': ['whole_brain', 'brainstem', 'thalamus'],
        'grid': 'brain_grid_dual',
        'controller': 'mycelial_network',
        'controller_description': 'Controls sleep through circadian and homeostatic networks',
        'tools': ['sleep_monitor', 'memory_consolidator', 'recovery_optimizer'],
        'models': ['biomimetic_sleep_model', 'circadian_rhythm_model', 'memory_consolidation_model'],
        'agents': ['sleep_agent', 'recovery_agent', 'circadian_agent'],
        'algorithms': ['sleep_regulation', 'memory_consolidation', 'recovery_optimization'],
        'triggers': ['circadian_signal', 'fatigue_detected', 'recovery_needed'],
        'brain_states': ['sleep_onset', 'light_sleep_n1', 'light_sleep_n2', 'deep_sleep_n3', 'rem_light', 'rem_intense'],
        'associated_brain_state_ids': [7, 8, 9, 10, 11, 12]
    },
    
    # Primitive and Instinctual States
    'primordial': {
        'cognitive_state_id': 25,
        'cognitive_state_name': 'primordial',
        'cognitive_state_description': 'Basic survival instincts and primitive responses',
        'brain_wave_frequency': (20, 40),  # High frequency emergency response
        'brain_wave_amplitude': (5, 20),  # Microvolts
        'brain_wave_pattern': 'survival_alertness',
        'brain_sub_region_involvement': ['brainstem', 'limbic_system', 'autonomic_centers'],
        'grid': 'mirror_grid',
        'controller': 'mycelial_network',
        'controller_description': 'Controls primordial responses through survival networks',
        'tools': ['threat_detector', 'survival_optimizer', 'instinct_processor'],
        'models': ['survival_model', 'instinct_model'],
        'agents': ['survival_agent', 'instinct_agent'],
        'algorithms': ['threat_assessment', 'survival_optimization', 'instinctual_response'],
        'triggers': ['threat_detected', 'survival_endangered', 'fight_or_flight'],
        'brain_states': ['hyper_alert'],
        'associated_brain_state_ids': [1]
    }
}

