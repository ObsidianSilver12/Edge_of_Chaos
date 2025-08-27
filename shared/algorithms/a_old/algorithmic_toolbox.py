"""
Algorithmic Toolbox for the Edge of Chaos System (V2)

This module provides a detailed, granular catalog of fundamental mathematical,
statistical, and signal processing algorithms. It is designed to serve as a
"toolbox" from which the main model can select specific functions for its
processing pipeline (Raw -> Patterns -> Fragments -> Nodes -> Map).

This approach separates pure algorithms from complex learning architectures,
aligning with the system's biomimetic and efficient design philosophy.
"""

from typing import Dict, Any

# =============================================================================
# 1. SIGNAL PROCESSING ALGORITHMS (Physics-Based Feature Extraction)
# Use Case: Raw Sensory -> Sensory Patterns
# =============================================================================
SIGNAL_PROCESSING = {
    "description": "Pure mathematical transforms for converting raw signals into feature domains.",
    "FFT": {
        "name": "Fast Fourier Transform",
        "type": "frequency_domain_transform",
        "function": "Decomposes a time-domain signal into its constituent frequencies.",
        "why_it_fits": "Biomimetic (mimics the cochlea). A fundamental, assumption-free way to analyze wave-based data.",
        "modalities": ["auditory", "metaphysical", "physical_state_vibrations"]
    },
    "CWT": {
        "name": "Continuous Wavelet Transform",
        "type": "time_frequency_analysis",
        "function": "Decomposes a signal into wavelets, capturing both frequency and precise temporal location.",
        "why_it_fits": "Superior to FFT for transient, non-stationary events. Crucial for a brain processing a dynamic reality.",
        "modalities": ["auditory", "emotional_state", "temporal"]
    },
    "AUTOCORRELATION": {
        "name": "Autocorrelation",
        "type": "periodicity_detection",
        "function": "Measures the similarity of a signal with a delayed copy of itself.",
        "why_it_fits": "An efficient, model-free way to find repeating patterns, pitch, or rhythm.",
        "modalities": ["auditory", "temporal", "metaphysical"]
    }
}

# =============================================================================
# 2. STATISTICAL & INFORMATION-THEORETIC ALGORITHMS
# Use Case: Pattern Evaluation, Significance Measurement, Correlation
# =============================================================================
STATISTICAL_ANALYSIS = {
    "description": "Algorithms for measuring relationships, information content, and significance.",
    "PEARSON_CORRELATION": {
        "name": "Pearson Correlation Coefficient",
        "type": "linear_correlation",
        "function": "Measures the linear relationship between two continuous variables (value from -1 to 1).",
        "why_it_fits": "A fast, baseline method for finding simple relationships between any two metrics.",
        "stage": ["Pattern Evaluation", "Fragments -> Nodes"]
    },
    "MUTUAL_INFORMATION": {
        "name": "Mutual Information",
        "type": "nonlinear_dependency",
        "function": "Measures the mutual dependence between two variables, capturing non-linear relationships.",
        "why_it_fits": "More powerful than Pearson for finding complex, hidden relationships between modalities.",
        "stage": ["Pattern Evaluation", "Fragments -> Nodes"]
    },
    "SHANNON_ENTROPY": {
        "name": "Shannon Entropy",
        "type": "information_content_measure",
        "function": "Measures the uncertainty or 'surprise' in a data source.",
        "why_it_fits": "The mathematical core of the 'Edge of Chaos' principle. Allows the model to score and prioritize patterns with optimal information content.",
        "stage": ["Pattern Evaluation"]
    },
    "GRANGER_CAUSALITY": {
        "name": "Granger Causality Test",
        "type": "predictive_relationship_test",
        "function": "A statistical test to determine if one time series is useful in forecasting another.",
        "why_it_fits": "Moves beyond correlation to predictive causality, essential for building a true world model.",
        "stage": ["Fragments -> Nodes", "Semantic World Map"]
    }
}

# =============================================================================
# 3. CLUSTERING & CLASSIFICATION ALGORITHMS (Unsupervised Grouping)
# Use Case: Sensory Patterns -> Fragments
# =============================================================================
CLUSTERING = {
    "description": "Algorithms for grouping similar patterns together without predefined labels.",
    "DBSCAN": {
        "name": "Density-Based Spatial Clustering of Applications with Noise",
        "type": "density_based_clustering",
        "function": "Groups points that are closely packed, marking low-density points as outliers.",
        "why_it_fits": "Biomimetic (finds 'natural' clusters) and excellent for novelty/outlier detection, which is critical for learning.",
        "stage": ["Sensory Patterns -> Fragments"]
    },
    "SOM": {
        "name": "Self-Organizing Map",
        "type": "competitive_learning_network",
        "function": "Produces a low-dimensional, discretized map of the input space, preserving topological properties.",
        "why_it_fits": "Extremely biomimetic, mimics the formation of cortical maps. Perfect for the spatial organization of the Semantic World Map.",
        "stage": ["Fragments -> Nodes", "Semantic World Map"]
    }
}

# =============================================================================
# 4. GRAPH & VECTOR SPACE ALGORITHMS
# Use Case: Text Analysis, Search, Associative Thought
# =============================================================================
GRAPH_AND_VECTOR = {
    "description": "Algorithms for operating on conceptual graphs and vector embeddings.",
    "TEXT_RANK": {
        "name": "TextRank (based on PageRank)",
        "type": "graph_based_ranking",
        "function": "Finds the most 'important' or 'central' nodes in a graph.",
        "why_it_fits": "Operates on a graph of concepts, not tokens. Ideal for summarization and keyword extraction.",
        "stage": ["Text Processing"]
    },
    "COSINE_SIMILARITY": {
        "name": "Cosine Similarity",
        "type": "vector_orientation_measure",
        "function": "Measures the cosine of the angle between two vectors to determine their directional similarity.",
        "why_it_fits": "The standard for fast, efficient semantic search in the memory base.",
        "stage": ["Search", "Correlation"]
    },
    "DIJKSTRA_A_STAR": {
        "name": "Dijkstra's Algorithm / A*",
        "type": "graph_pathfinding",
        "function": "Finds the shortest path between two nodes in a weighted graph.",
        "why_it_fits": "Enables biomimetic 'trains of thought' by finding the most direct associative path between concepts.",
        "stage": ["Search", "Reasoning"]
    }
}

# =============================================================================
# 5. LEARNING ARCHITECTURES & OBJECTIVES (Methodologies)
# Use Case: Creating the abstract vector representations for the above algorithms to use.
# =============================================================================
LEARNING_ARCHITECTURES = {
    "description": "These are not single algorithms, but training methodologies that combine encoders and loss functions to learn representations.",
    "JOINT_EMBEDDING_PREDICTIVE": {
        "name": "Joint-Embedding Predictive Architecture (JEPA)",
        "core_idea": "Learn abstract representations by predicting the embeddings of masked-out data from visible data.",
        "underlying_algorithms": ["Neural Network Encoder (e.g., ViT, CNN)", "Masking Strategy", "Predictive Loss Function (e.g., Cosine Similarity Loss)"],
        "why_it_fits": "The training process is a powerful way to generate the high-quality, semantic vectors your other algorithms need, without costly data reconstruction."
    },
    "CONTRASTIVE_LEARNING": {
        "name": "Contrastive Learning (e.g., SimCLR)",
        "core_idea": "Learn representations by pulling augmented 'positive' pairs of a sample closer together in the embedding space, while pushing 'negative' pairs apart.",
        "underlying_algorithms": ["Data Augmentation", "Neural Network Encoder", "Contrastive Loss Function (e.g., NT-Xent)"],
        "why_it_fits": "An alternative to JEPA for learning powerful representations. Very effective but requires careful selection of augmentations and negative samples."
    }
}