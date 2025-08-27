"""
Algorithm Catalog for the Edge of Chaos System

This module serves as a central repository for the various algorithms used
across the data processing pipeline (Raw Sensory -> Patterns -> Nodes -> World Map).
It provides a structured way to define, group, and select algorithms for
different tasks and sensory modalities, acting as the initial "toolbox" for the
biomimetic learning model.
"""

from typing import Dict, List, Any

# =============================================================================
# 1. PATTERN DETECTION ALGORITHMS (Multimodal)
# =============================================================================
PATTERN_DETECTION = {
    "description": "Algorithms for identifying patterns. Grouped into fundamental signal processing and higher-level abstract learning.",
    "signal_processing": {
        "FFT": {
            "name": "Fast Fourier Transform",
            "type": "frequency_analysis",
            "use_case": "Decomposing any signal (auditory, metaphysical, temporal) into its constituent frequencies. Pure physics-based feature extraction.",
            "modalities": ["auditory", "metaphysical", "temporal", "physical_state"]
        },
        "WAVELET_TRANSFORM": {
            "name": "Continuous Wavelet Transform (CWT)",
            "type": "time_frequency_analysis",
            "use_case": "Analyzing signals where frequency content changes over time. Superior to FFT for transient events, audio clicks, or non-stationary data.",
            "modalities": ["auditory", "temporal", "emotional_state"]
        },
        "AUTOCORRELATION": {
            "name": "Autocorrelation",
            "type": "periodicity_detection",
            "use_case": "Finding repeating patterns or fundamental frequencies within a single continuous signal stream.",
            "modalities": ["auditory", "temporal", "metaphysical"]
        }
    },
    "abstract_learning": {
        "JEPA": {
            "name": "Joint-Embedding Predictive Architecture",
            "type": "predictive_coding",
            "use_case": "Learning abstract representations from any modality by predicting missing information in representation space. Ideal for semantic pattern detection.",
            "modalities": ["visual", "auditory", "text", "metaphysical"],
            "notes": "A powerful training technique for creating high-quality semantic embeddings from processed features."
        },
        "MAMBA": {
            "name": "Selective State Space Model",
            "type": "sequence_modeling",
            "use_case": "Processing long, continuous sequences like raw audio, text streams, or temporal data with linear complexity.",
            "modalities": ["auditory", "text", "temporal"],
            "notes": "Excellent alternative to Transformers for non-tokenized sequential data."
        }
    }
}

# =============================================================================
# 2. DATA CLASSIFICATION ALGORITHMS
# =============================================================================
DATA_CLASSIFICATION = {
    "description": "Algorithms for categorizing data based on learned features.",
    "supervised": {
        "LINEAR_CLASSIFIER": {
            "name": "Linear Classifier / Simple Feed-Forward Network",
            "use_case": "Efficiently classifying the abstract representation vectors produced by JEPA.",
            "notes": "To be used on top of a powerful feature extractor."
        }
    },
    "unsupervised": {
        "SOM": {
            "name": "Self-Organizing Map",
            "type": "clustering",
            "use_case": "Topological clustering of sensory data to find emergent categories. Highly biomimetic, mimics cortical maps.",
            "modalities": ["visual", "auditory", "metaphysical"]
        },
        "DBSCAN": {
            "name": "Density-Based Spatial Clustering of Applications with Noise",
            "type": "clustering",
            "use_case": "Clustering data based on density, effective at identifying outliers and non-linear shapes.",
            "modalities": ["spatial", "visual"]
        }
    }
}

# =============================================================================
# 3. TEXT EXTRACTION ALGORITHMS
# =============================================================================
TEXT_EXTRACTION = {
    "description": "Algorithms for extracting summaries and labels from text without traditional tokenization.",
    "labeling": {
        "TEXT_RANK": {
            "name": "TextRank",
            "type": "graph_based",
            "use_case": "Identifying key concepts by treating them as nodes in a graph and finding the most central ones.",
            "notes": "Works on extracted concepts, not raw tokens."
        }
    },
    "summarization": {
        "EMBEDDING_CLUSTERING": {
            "name": "Extractive Summarization via Sentence Embedding Clustering",
            "type": "extractive",
            "use_case": "Generate a vector for each sentence (using Mamba or JEPA), cluster them, and select the most representative sentence from each cluster.",
            "notes": "Avoids generative models, focusing on extracting key information."
        }
    }
}

# =============================================================================
# 4. MEMORY & SEARCH ALGORITHMS
# =============================================================================
MEMORY_AND_SEARCH = {
    "description": "Algorithms for building and querying the memory base.",
    "memory_creation": {
        "DENDRITIC_INTEGRATION": {
            "name": "Dendritic Computing Model",
            "use_case": "Integrating fragments and patterns into a single node, using scoring and correlation logic to form a coherent concept.",
            "notes": "The core logic for how a 'Node' is formed in the brain structure."
        },
        "MYCELIAL_GROWTH": {
            "name": "Mycelial Network Growth",
            "use_case": "Dynamically forming and strengthening connections between nodes based on resonance, semantic similarity, and frequency coherence.",
            "notes": "Creates an associative, self-organizing memory graph. Should incorporate 'preferential attachment' to strengthen connections within a knowledge domain."
        }
    },
    "additive_thought": {
        "NODE_ACTIVATION_DRILLDOWN": {
            "name": "Node Activation Drill-down",
            "type": "retrieval_synthesis",
            "use_case": "When a node is activated, this process retrieves its source SENSORY_RAW data and can trigger on-demand summarization or detail-generation using specialized tools (e.g., small LMs).",
            "notes": "This is the mechanism for 'additive thought' - going from a general concept to its specific details."
        }
    },
    "search": {
        "VECTOR_SIMILARITY": {
            "name": "Vector Similarity Search",
            "type": "semantic_search",
            "use_case": "Finding semantically related nodes/fragments by comparing their JEPA-generated vector embeddings using Cosine Similarity.",
            "notes": "The primary method for fast, meaning-based retrieval."
        },
        "GRAPH_TRAVERSAL": {
            "name": "Graph Traversal Algorithms",
            "type": "associative_search",
            "use_case": "Finding paths and relationships between concepts in the mycelial network (e.g., A* for pathfinding, PageRank for importance).",
            "notes": "Enables associative 'trains of thought'."
        },
        "RESONANCE_SEARCH": {
            "name": "Frequency Resonance Search",
            "type": "metaphysical_search",
            "use_case": "Finding nodes that resonate with a given query based on frequency, color, and elemental properties.",
            "notes": "Ideal for emotional, spiritual, or abstract queries."
        }
    }
}

# =============================================================================
# 5. SENSORY ANALYSIS ALGORITHMS
# =============================================================================
SENSORY_ANALYSIS = {
    "description": "Algorithms for statistical analysis, prediction, and significance measurement for each sensory type.",
    "visual": {
        "correlation": "Spatial Cross-Correlation",
        "prediction": "V-JEPA, Diffusion Models",
        "significance": "Saliency Maps from Vision Transformer Attention"
    },
    "auditory": {
        "correlation": "Autocorrelation, Spectral Coherence",
        "prediction": "Mamba, Spiking Neural Networks (SNNs)",
        "significance": "Psychoacoustic Loudness and Sharpness Models"
    },
    "text": {
        "correlation": "Semantic Similarity (Vector-based), N-gram analysis on concepts",
        "prediction": "Mamba (Next-character/concept)",
        "significance": "TextRank on Concept Graph"
    },
    "metaphysical": {
        "correlation": "Harmonic Analysis, Frequency Resonance Calculation",
        "prediction": "Kalman Filters on awareness/energy time-series",
        "significance": "Coherence, Intensity, and Resonance with Core Identity"
    }
}

# =============================================================================
# 6. PATTERN EVALUATION & VALIDATION ALGORITHMS
# =============================================================================
PATTERN_EVALUATION = {
    "description": "A suite of algorithms to evaluate the quality of a learned pattern BEFORE it is integrated into a node. This is the 'critical thinking' layer.",
    "consistency": {
        "CROSS_MODAL_CONSISTENCY": {
            "name": "Cross-Modal Consistency Check",
            "use_case": "Takes vector representations from different modalities (e.g., visual and auditory) captured at the same time and calculates their semantic similarity. High similarity indicates a robust, consistent pattern.",
            "notes": "Ensures that what is seen aligns with what is heard, etc."
        }
    },
    "information_content": {
        "SHANNON_ENTROPY": {
            "name": "Information Entropy",
            "use_case": "Measures the unpredictability or information content of a pattern. The model can learn to prioritize patterns with an optimal amount of entropy (not too simple, not too random).",
            "notes": "Directly relates to the 'Edge of Chaos' principle."
        },
        "LEMPEL_ZIV_COMPLEXITY": {
            "name": "Lempel-Ziv Complexity",
            "use_case": "Measures the algorithmic complexity of a pattern by assessing its compressibility. Truly complex patterns are not easily compressible.",
            "notes": "A good way to distinguish complex, meaningful patterns from simple repetitive ones or random noise."
        }
    },
    "predictive_quality": {
        "PREDICTIVE_ACCURACY_SCORE": {
            "name": "Predictive Accuracy Score",
            "use_case": "If a pattern was generated by a predictive model (like JEPA or Mamba), this evaluates how well the model's predictions matched the actual subsequent data. High accuracy indicates a valid, useful pattern.",
            "notes": "A feedback mechanism to reward models that learn the true underlying dynamics of the data."
        }
    }
}

# =============================================================================
# 7. BIOLOGICALLY-INSPIRED & EXPERIMENTAL ALGORITHMS
# =============================================================================
EXPERIMENTAL = {
    "description": "Advanced or experimental algorithms that the model could learn to use over time for specialized tasks.",
    "temporal_processing": {
        "SNN": {
            "name": "Spiking Neural Network",
            "type": "event_based_processing",
            "use_case": "Detecting precise temporal patterns using Spike-Timing-Dependent Plasticity (STDP). Highly energy-efficient and event-driven.",
            "modalities": ["auditory", "temporal", "metaphysical"],
            "notes": "A potential future optimization for the model to adopt if it proves more efficient for certain tasks than continuous models."
        }
    }
}
