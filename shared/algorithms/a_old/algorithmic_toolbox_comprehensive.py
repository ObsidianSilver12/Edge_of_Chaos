"""
Comprehensive Algorithmic Toolbox for the Edge of Chaos System

This module provides a detailed, granular catalog of fundamental mathematical,
statistical, and signal processing algorithms. It is designed to serve as a
comprehensive "toolbox" from which the model can select specific functions for
its processing pipeline (Raw Sensory -> Patterns -> Fragments -> Nodes -> Map).

This approach separates pure algorithms from complex learning architectures,
aligning with the system's biomimetic and efficient design philosophy.

--- SCORING SYSTEM ---
Each algorithm is rated on three scales from 1 to 5:

- Fit Score (1-5): How biomimetic and aligned with the project's philosophy.
  (5 = Perfectly aligned, physics-based, non-tokenizing; 1 = Traditional, black-box).

- Complexity Score (1-5): How complex the algorithm is to implement and understand.
  (1 = Trivial, e.g., a simple sum; 5 = Highly complex, e.g., a novel architecture).

- Compute Score (1-5): How computationally expensive the algorithm is.
  (1 = Very fast/cheap, e.g., O(n); 5 = Very slow/expensive, e.g., O(n^3) or worse).

"""

from typing import Dict, List, Any

ALGORITHMIC_TOOLBOX: Dict[str, Dict[str, Any]] = {
    # =============================================================================
    # USE CASE 1: FUNDAMENTAL SIGNAL PROCESSING & FEATURE EXTRACTION
    # For converting raw sensory input into structured mathematical features.
    # =============================================================================
    "FUNDAMENTAL_SIGNAL_PROCESSING": {
        "description": "Core algorithms for decomposing raw signals and images into their fundamental mathematical components. This is the first layer of unbiased feature extraction.",
        "algorithms": [
            {
                "name": "Fast Fourier Transform (FFT)",
                "description": "A highly efficient algorithm that decomposes a signal (e.g., audio) into its constituent sine and cosine waves, revealing its frequency spectrum.",
                "fit_score": 5,
                "complexity_score": 2,
                "compute_score": 2,
                "applicable_senses": ["auditory", "metaphysical", "physical_state", "temporal"]
            },
            {
                "name": "Continuous Wavelet Transform (CWT)",
                "description": "A time-frequency analysis tool that captures not just *what* frequencies are present, but precisely *when* they occur. Ideal for transient, non-stationary signals.",
                "fit_score": 5,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["auditory", "emotional_state", "temporal", "metaphysical"]
            },
            {
                "name": "Mel-Frequency Cepstral Coefficients (MFCCs)",
                "description": "Creates a representation of a sound's power spectrum that is based on the non-linear frequency scale of human hearing.",
                "fit_score": 4,
                "complexity_score": 3,
                "compute_score": 3,
                "applicable_senses": ["auditory", "text"]
            },
            {
                "name": "Edge Detection (Canny, Sobel, Laplacian)",
                "description": "Identifies points in an image where brightness changes sharply, corresponding to edges. This mimics the function of V1/V2 neurons in the visual cortex.",
                "fit_score": 5,
                "complexity_score": 2,
                "compute_score": 2,
                "applicable_senses": ["visual"]
            },
            {
                "name": "Saliency Maps (Itti-Koch-Niebur Model)",
                "description": "A purely computational model that predicts which parts of an image a human would focus on, based on features like color, intensity, and orientation.",
                "fit_score": 5,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["visual"]
            },
            {
                "name": "Optical Flow (Lucas-Kanade, Horn-Schunck)",
                "description": "Calculates the motion of objects between consecutive frames of a video. It estimates the velocity vectors of pixels.",
                "fit_score": 5,
                "complexity_score": 3,
                "compute_score": 3,
                "applicable_senses": ["visual"]
            }
        ]
    },

    # =============================================================================
    # USE CASE 2: STATISTICAL ANALYSIS & CORRELATION
    # For measuring relationships between features and patterns.
    # =============================================================================
    "STATISTICAL_ANALYSIS": {
        "description": "Algorithms for quantifying relationships, dependencies, and predictive power between different data streams and features.",
        "algorithms": [
            {
                "name": "Pearson & Spearman Correlation",
                "description": "Pearson measures the linear relationship between two variables. Spearman measures the monotonic relationship (robust to outliers and non-linearities).",
                "fit_score": 4,
                "complexity_score": 1,
                "compute_score": 1,
                "applicable_senses": ["visual", "auditory", "text", "emotional_state", "physical_state", "spatial", "temporal", "metaphysical", "algorithmic", "other_data"]
            },
            {
                "name": "Autocorrelation & Cross-Correlation",
                "description": "Autocorrelation finds repeating patterns in a single signal. Cross-correlation finds time-delayed similarities between two different signals.",
                "fit_score": 5,
                "complexity_score": 2,
                "compute_score": 2,
                "applicable_senses": ["auditory", "temporal", "metaphysical", "visual"]
            },
            {
                "name": "Granger Causality",
                "description": "A statistical test to determine if one time series is useful in forecasting another. Moves beyond correlation to predictive relationships.",
                "fit_score": 4,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["temporal", "emotional_state", "physical_state", "metaphysical", "algorithmic"]
            },
            {
                "name": "Canonical Correlation Analysis (CCA)",
                "description": "A method for finding the relationships between two sets of variables. Useful for finding shared information between two different sensory modalities (e.g., visual and auditory).",
                "fit_score": 3,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["visual", "auditory", "text", "emotional_state"]
            }
        ]
    },

    # =============================================================================
    # USE CASE 3: PATTERN RECOGNITION & CLUSTERING
    # For grouping similar patterns into fragments without predefined labels.
    # =============================================================================
    "PATTERN_RECOGNITION_AND_CLUSTERING": {
        "description": "Unsupervised algorithms for discovering natural groupings and structures within the feature data, forming the basis of 'fragments'.",
        "algorithms": [
            {
                "name": "DBSCAN (Density-Based Clustering)",
                "description": "Groups together points that are closely packed, marking outliers. Finds natural clusters of arbitrary shape and doesn't require knowing the number of clusters beforehand.",
                "fit_score": 5,
                "complexity_score": 3,
                "compute_score": 3,
                "applicable_senses": ["spatial", "visual", "auditory", "text"]
            },
            {
                "name": "Self-Organizing Maps (SOMs)",
                "description": "A type of neural network that produces a low-dimensional 'map' of the input space, preserving topological properties. Mimics the formation of cortical maps in the brain.",
                "fit_score": 5,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["visual", "auditory", "metaphysical", "text"]
            },
            {
                "name": "Dynamic Time Warping (DTW)",
                "description": "Finds the optimal alignment between two time-varying sequences. Excellent for comparing temporal patterns like speech, gestures, or emotional trajectories.",
                "fit_score": 5,
                "complexity_score": 3,
                "compute_score": 4,
                "applicable_senses": ["auditory", "temporal", "emotional_state", "text"]
            },
            {
                "name": "Independent Component Analysis (ICA)",
                "description": "A computational method for separating a multivariate signal into additive, non-Gaussian subcomponents. Excellent for 'blind source separation', like isolating individual voices from a mixed audio signal.",
                "fit_score": 4,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["auditory", "visual", "metaphysical"]
            }
        ]
    },

    # =============================================================================
    # USE CASE 4: TEMPORAL & SEQUENCE ANALYSIS
    # For processing data that evolves over time.
    # =============================================================================
    "TEMPORAL_AND_SEQUENCE_ANALYSIS": {
        "description": "Algorithms designed specifically for handling sequential data, making predictions, and understanding time-based dynamics.",
        "algorithms": [
            {
                "name": "Kalman Filter",
                "description": "An optimal estimation algorithm that uses a series of measurements observed over time to produce estimates of unknown variables. Perfect for tracking and predicting the state of a dynamic system.",
                "fit_score": 5,
                "complexity_score": 3,
                "compute_score": 2,
                "applicable_senses": ["physical_state", "spatial", "temporal", "metaphysical", "algorithmic"]
            },
            {
                "name": "Hidden Markov Models (HMMs)",
                "description": "A statistical model for sequences with unobserved (hidden) states. Classic tool for modeling systems like speech, where phonemes are the hidden states.",
                "fit_score": 4,
                "complexity_score": 4,
                "compute_score": 3,
                "applicable_senses": ["auditory", "text", "temporal"]
            },
            {
                "name": "Conditional Random Fields (CRFs)",
                "description": "A classical statistical modeling method for structured prediction. Unlike HMMs, they don't assume independence between features, making them more powerful for tasks like sequence labeling.",
                "fit_score": 3,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["text", "auditory"]
            }
        ]
    },

    # =============================================================================
    # USE CASE 5: TOPOLOGICAL & GRAPH ANALYSIS
    # For analyzing the structure of the semantic map and conceptual relationships.
    # =============================================================================
    "TOPOLOGICAL_AND_GRAPH_ANALYSIS": {
        "description": "Algorithms that operate on graphs, ideal for analyzing the final semantic world map, finding relationships between nodes, and enabling associative thought.",
        "algorithms": [
            {
                "name": "PageRank / TextRank",
                "description": "A graph-based ranking algorithm to measure the importance of nodes. TextRank applies this to text, treating concepts or sentences as nodes.",
                "fit_score": 5,
                "complexity_score": 2,
                "compute_score": 2,
                "applicable_senses": ["text", "algorithmic"]
            },
            {
                "name": "Dijkstra's Algorithm / A*",
                "description": "Graph traversal algorithms to find the shortest (most efficient) path between two nodes. Models an associative 'train of thought'.",
                "fit_score": 5,
                "complexity_score": 2,
                "compute_score": 2,
                "applicable_senses": ["algorithmic", "spatial"]
            },
            {
                "name": "Louvain Modularity (Community Detection)",
                "description": "An algorithm for detecting communities or modules in large networks. Can be used to discover high-level knowledge domains in the semantic map.",
                "fit_score": 4,
                "complexity_score": 4,
                "compute_score": 3,
                "applicable_senses": ["algorithmic", "text"]
            }
        ]
    },

    # =============================================================================
    # USE CASE 6: INFORMATION THEORETIC MEASURES
    # For evaluating the quality, complexity, and novelty of learned patterns.
    # =============================================================================
    "INFORMATION_THEORETIC_MEASURES": {
        "description": "Pure mathematical measures to evaluate the quality of learned patterns, aligning with the 'Edge of Chaos' philosophy. This is the system's 'critical thinking' layer.",
        "algorithms": [
            {
                "name": "Shannon Entropy",
                "description": "Measures the uncertainty, surprise, or 'information content' of a data source. The model can learn to prioritize patterns with optimal (not too high, not too low) entropy.",
                "fit_score": 5,
                "complexity_score": 1,
                "compute_score": 1,
                "applicable_senses": ["visual", "auditory", "text", "emotional_state", "physical_state", "spatial", "temporal", "metaphysical", "algorithmic", "other_data"]
            },
            {
                "name": "Mutual Information",
                "description": "Measures the mutual dependence between two variables, capturing non-linear relationships. A powerful tool for validating cross-modal consistency (e.g., do the visual and audio streams contain related information?).",
                "fit_score": 5,
                "complexity_score": 3,
                "compute_score": 3,
                "applicable_senses": ["visual", "auditory", "text", "emotional_state", "physical_state", "spatial", "temporal", "metaphysical", "algorithmic", "other_data"]
            },
            {
                "name": "Lempel-Ziv Complexity",
                "description": "Measures the algorithmic complexity of a pattern by assessing its compressibility. Distinguishes meaningful, complex patterns from both simple repetition and random noise.",
                "fit_score": 5,
                "complexity_score": 3,
                "compute_score": 2,
                "applicable_senses": ["visual", "auditory", "text", "temporal", "algorithmic"]
            }
        ]
    }
}

ALGORITHMIC_TOOLBOX.update({
    # =============================================================================
    # USE CASE 7: TEXT & NATURAL LANGUAGE PROCESSING (Non-Tokenizing)
    # For extracting meaning and structure from text data.
    # =============================================================================
    "TEXT_AND_NLP": {
        "description": "Algorithms for analyzing text by focusing on concepts, sentences, and statistical properties rather than traditional tokenization.",
        "algorithms": [
            {
                "name": "TF-IDF (Term Frequency-Inverse Document Frequency)",
                "description": "A numerical statistic that reflects how important a word (or concept) is to a document in a collection or corpus. Good for keyword extraction.",
                "fit_score": 3,
                "complexity_score": 2,
                "compute_score": 2,
                "applicable_senses": ["text"]
            },
            {
                "name": "Latent Dirichlet Allocation (LDA)",
                "description": "A generative statistical model for discovering abstract 'topics' that occur in a collection of documents. Can be applied to your extracted concepts to find thematic clusters.",
                "fit_score": 4,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["text"]
            },
            {
                "name": "Concept Vector Models (e.g., Word2Vec/GloVe adapted)",
                "description": "Learns vector representations for words/concepts based on their context. Instead of words, you can train it on your extracted `key_concepts` to learn their relationships.",
                "fit_score": 4,
                "complexity_score": 3,
                "compute_score": 4,
                "applicable_senses": ["text", "algorithmic"]
            }
        ]
    },

    # =============================================================================
    # USE CASE 8: DIMENSIONALITY REDUCTION & MANIFOLD LEARNING
    # For understanding and visualizing high-dimensional data (like embeddings).
    # =============================================================================
    "DIMENSIONALITY_REDUCTION": {
        "description": "Algorithms to reduce the number of variables in a dataset while preserving important information. Essential for visualizing and interpreting the high-dimensional vectors from JEPA-like models.",
        "algorithms": [
            {
                "name": "Principal Component Analysis (PCA)",
                "description": "A linear technique that transforms data into a new coordinate system, ordering dimensions by the amount of variance they explain. Fast and excellent for data pre-processing.",
                "fit_score": 4,
                "complexity_score": 2,
                "compute_score": 3,
                "applicable_senses": ["visual", "auditory", "text", "metaphysical", "algorithmic"]
            },
            {
                "name": "t-SNE (t-Distributed Stochastic Neighbor Embedding)",
                "description": "A non-linear technique particularly well-suited for visualizing high-dimensional datasets. It models data points by their similarity, revealing underlying cluster structures.",
                "fit_score": 3,
                "complexity_score": 4,
                "compute_score": 5,
                "applicable_senses": ["visual", "auditory", "text", "metaphysical", "algorithmic"]
            },
            {
                "name": "UMAP (Uniform Manifold Approximation and Projection)",
                "description": "A modern non-linear technique that is often faster than t-SNE and better at preserving the global structure of the data. Excellent for finding the underlying 'shape' of your data.",
                "fit_score": 4,
                "complexity_score": 4,
                "compute_score": 4,
                "applicable_senses": ["visual", "auditory", "text", "metaphysical", "algorithmic"]
            }
        ]
    },

