Excellent. Now, let's build the comprehensive catalog for the **Visual Sense (Video)**.

Processing video presents a significant step up in complexity from still images, primarily due to the introduction of the **temporal dimension**. While many foundational image processing functions are still relevant on a per-frame basis, the true essence of video analysis lies in understanding motion, change, and the narrative that unfolds over time.

This catalog is designed to give you a full toolkit for capturing that temporal essence.

---

### **Comprehensive Algorithmic Catalog: Visual Sense (Video)**

This catalog outlines the tools for processing video data, from analyzing raw frame sequences to understanding complex events and integrating them into the Semantic World Map.

#### **Stage 1: SENSORY_RAW - Foundational Video Feature Extraction**

**Goal:** To convert a raw video stream (a sequence of images) into a structured object containing objective, low-level features that describe both the spatial content of frames and the temporal changes between them. This populates the video-specific sections of the `SENSORY_RAW['visual']` dictionary, especially `motion_properties`.

##### **Group 1.1: Per-Frame Spatial Functions**
*(These are the same image functions from the previous catalog, applied to individual or keyframes of the video)*

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Keyframe Selection** | Identifies the most representative frames in a video to reduce redundant processing. | Function/Formula | `sensing` | 4 | 2 | 5 |
| **All Image Functions** | Color Histograms, Edge Detection, Corner Detection, Texture Analysis (Gabor, LBP), Feature Descriptors (SIFT, ORB) are all applied to selected keyframes. | Function/Formula | `sensing` | (Varies) | (Varies) | 4 |

##### **Group 1.2: Foundational Temporal & Motion Functions**
*(These functions are the core of video analysis, capturing change and movement)*

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frame Differencing** | A simple method to detect motion by subtracting consecutive frames. Highly sensitive to noise but computationally cheap. | Function/Formula | `sensing` | 4 | 1 | 5 |
| **Optical Flow (Dense & Sparse)**| **Critical Function:** Calculates a vector field showing the direction and magnitude of motion for every pixel (dense) or for specific feature points (sparse) between frames. Directly mimics motion processing in the visual cortex. | Function/Formula | `sensing` | **5** | 3 | 3 |
| **Motion History Image (MHI)** | Creates a grayscale image where pixel intensity represents the recency of motion. Provides a simple, powerful summary of movement over a short time window. | Function/Formula | `sensing`, `remembering`| 5 | 2 | 5 |
| **Background Subtraction** | Models the static background of a scene to isolate moving foreground objects (the "actors"). | Methodology | `sensing`, `thinking`| 4 | 3 | 4 |
| **3D Convolutions** | An extension of 2D image convolutions that also operates across the time dimension, allowing a neural network to directly learn spatiotemporal features from raw pixel data. | Model Component | `sensing` | 3 | 3 | 3 |

---

#### **Stage 2: SENSORY_PATTERNS - Intra-Video Pattern Discovery**

**Goal:** To analyze the foundational features to discover meaningful patterns of movement, object interactions, and scene changes over time. Populates `SENSORY_PATTERNS` with rich temporal and motion-based information.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Object Tracking (Kalman Filter, Particle Filter)**| Uses motion information from optical flow or background subtraction to maintain the identity and trajectory of a specific object across many frames. | Methodology | `sensing`, `remembering`| 5 | 4 | 4 |
| **Trajectory Clustering** | Groups the movement paths (trajectories) of multiple objects to find common patterns of movement (e.g., cars all turning left at an intersection). | Function/Formula | `conceptualising` | 4 | 3 | 4 |
| **Spatiotemporal Interest Points (STIPs)**| An extension of image keypoints (like SIFT) to the time dimension. Finds salient points in both space and time, such as the peak of an action. | Methodology | `sensing` | 4 | 4 | 3 |
| **Action Primitives Recognition**| Recognizes simple, short actions by analyzing motion patterns (e.g., "walking," "running," "waving"). Often done with models like 3D-CNNs or LSTMs on optical flow data. | Model | `sensing`, `learning` | 4 | 4 | 3 |
| **Scene Change/Shot Boundary Detection**| Identifies cuts, fades, and dissolves in a video, segmenting it into coherent scenes. | Function/Formula | `sensing` | 4 | 2 | 5 |

---

#### **Stage 3: FRAGMENT - Consolidation and Coherent Event Generation**

**Goal:** To synthesize the video patterns into a unified, coherent representation within the multi-modal `FRAGMENT`, generating a narrative description of the events that occurred.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Event Graph Synthesis (GNN)**| **Core Methodology:** Constructs a "temporal scene graph" where tracked objects are nodes and their interactions over time (e.g., "person A picks up object B") become time-stamped, directed edges. A GNN learns a holistic representation of the entire event. | Methodology | `conceptualising`, `resolving`| 4 | 4 | 4 |
| **Video Captioning Models (as Tools)**| Uses a pre-trained model (e.g., a Video-Transformer) as a utility to generate the `scene_description` for the entire video clip. | Model (Tool) | `writing`, `thinking` | 2 | 3 | 3 |
| **Video Summarization** | A methodology to automatically select the most important keyframes or sub-clips that represent the entire video, creating a visual "abstract". | Methodology | `thinking` | 3 | 4 | 3 |
| **Causal Inference from Motion**| A reasoning process that analyzes object trajectories and interactions to infer causality (e.g., if ball A hits ball B, and then ball B moves, it infers A caused B to move). | Methodology | `reasoning` | 5 | 4 | 4 |

---

#### **Stage 4: NODE - Advanced Event Reasoning and Network Integration**

**Goal:** "Conscious" analysis of the video fragment. This involves deep contextual understanding of the event, validating it against the world model, and integrating it into the brain's network.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V-JEPA (Video Joint-Embedding Predictive Architecture)**| **Primary World Model:** The definitive architecture for this stage. It takes the video fragment and processes it, predicting future frames or masked-out sections in an abstract representation space. This directly tests the AI's understanding of physical dynamics and event plausibility. High success = high `consensus_score`. | Methodology / Arch.| `learning`, `reasoning`, `sensing` | **5** | 4 | 2 |
| **Video Transformers (ViViT, TimeSformer)**| **Feature Extractor:** These models are the core encoders used within a V-JEPA framework. They are designed to efficiently apply attention mechanisms across both space and time to capture complex spatiotemporal relationships. | Model / Arch. | `thinking` | 3 | 4 | 2 |
| **Graph Neural Networks (GNNs)**| **Relational Reasoner:** Performs `node-to-node matching`. It compares the new node's temporal scene graph against the entire brain's knowledge graph to find connections based on similar events, actors, or causal chains. | Model / Arch. | `reasoning` | 4 | 4 | 4 |
| **Coherence Validators** | **Dissonance Detector:** A suite of models/functions to check for event dissonance. For example, does the video violate the AI's learned model of physics? Does a person in the video perform an action that contradicts their known abilities in another node? | Methodology | `reasoning`, `resolving`| 5 | 4 | 4 |

---

#### **Stage 5: SEMANTIC_WORLD_MAP - Efficient Indexing for Retrieval**

**Goal:** To create a fast, searchable index of the validated node's video content for quick retrieval.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Video Embedding Models (VideoCLIP, etc.)**| **Semantic Indexer:** A pre-trained model used as a tool to create a single, dense vector embedding from the node's video clip. This allows for searching by video similarity ("find events that are similar to this one"). | Model (Tool) | `remembering`, `searching` | 3 | 3 | 3 |
| **Action/Event Tag Index (BM25)**| **Keyword Indexer:** The descriptive tags, recognized actions, and generated captions from the node are put into a classical sparse vector index for fast keyword-based retrieval of events. | Function/Formula | `remembering`, `searching` | 2 | 2 | 5 |
| **HNSW Index (FAISS, etc.)**| **Vector Database Engine:** The data structure that stores the video embeddings and allows for extremely fast similarity search. The core of "query-by-video." | Function/Formula | `remembering`, `searching` | 3 | 4 | 4 |