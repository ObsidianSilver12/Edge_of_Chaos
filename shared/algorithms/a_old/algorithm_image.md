Excellent point. Separating still images from video is crucial. They share foundational principles but diverge significantly in temporal analysis. A focused, comprehensive catalog for **Visual (Image) Sense** will ensure you have the right tools for each stage.

Let's build the definitive catalog for processing static visual information.

---

### **Comprehensive Algorithmic Catalog: Visual Sense (Still Image)**

This catalog outlines the algorithms, models, and methodologies for processing still images, from raw pixel data to their final indexed form in the Semantic World Map.

#### **Stage 1: SENSORY_RAW - Foundational Image Feature Extraction**

**Goal:** To convert a raw image into a structured object containing objective, low-level visual features. This is about deconstructing the image into its fundamental components of color, light, texture, and structure to populate the `SENSORY_RAW['visual']` dictionary.

##### **Group 1.1: Color & Spectral Functions**
*(Analyzing the distribution and properties of light and color)*

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Color Histogram Analysis** | Quantifies the distribution of colors (per channel R,G,B) and overall brightness. | Function/Formula | `sensing` | 4 | 1 | 5 |
| **Dominant Color Extraction** | Uses clustering (e.g., K-Means on pixels) to find the N most representative colors in the image, creating a palette. | Function/Formula | `sensing` | 3 | 2 | 5 |
| **Color Correlogram** | Measures the spatial correlation of color pairs. Captures how colors are distributed relative to each other. | Function/Formula | `sensing` | 4 | 3 | 4 |
| **White Balance / Color Temperature Estimation**| Analyzes the image to estimate the "warmth" or "coolness" of the ambient light. | Function/Formula | `sensing` | 4 | 2 | 5 |

##### **Group 1.2: Edge, Line & Corner Functions**
*(Identifying structural primitives, mimicking early visual cortex)*

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Operators (Sobel, Prewitt, Scharr)** | Calculate the directional intensity gradient at each pixel. The basis for all edge detection. | Function/Formula | `sensing` | 5 | 1 | 5 |
| **Canny Edge Detection** | A multi-stage algorithm that produces clean, thin edge maps by using non-maximum suppression and hysteresis thresholding. | Function/Formula | `sensing` | 5 | 3 | 5 |
| **Laplacian of Gaussian (LoG)**| Finds regions of rapid intensity change. Excellent for finding blobs and fine edges of all orientations. | Function/Formula | `sensing` | 5 | 3 | 4 |
| **Hough Transform (Lines, Circles)** | An algorithm for detecting instances of geometric shapes (lines, circles) by voting in a parameter space. | Function/Formula | `sensing` | 3 | 3 | 4 |
| **Harris & Shi-Tomasi Corner Detectors**| Identifies corner points as stable interest points for feature tracking and matching. | Function/Formula | `sensing` | 4 | 2 | 5 |

##### **Group 1.3: Texture & Pattern Functions**
*(Analyzing surface properties and repeating structures)*

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gabor Filter Bank** | A bank of filters at different orientations and frequencies. The premier method for texture analysis, directly mimicking V1 simple cells. | Function/Formula | `sensing` | **5** | 3 | 4 |
| **Local Binary Patterns (LBP)**| A powerful and computationally efficient texture descriptor based on local pixel comparisons. | Function/Formula | `sensing` | 4 | 2 | 5 |
| **Haralick Texture Features** | A suite of 14 statistical measures derived from the Gray-Level Co-occurrence Matrix (GLCM), describing texture properties like contrast and homogeneity. | Function/Formula | `sensing` | 3 | 3 | 4 |

##### **Group 1.4: Feature Point Detection & Description**
*(Finding and describing salient, invariant points for matching and recognition)*

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT (Scale-Invariant Feature Transform)**| Detects and describes local features that are invariant to scale and rotation. A canonical computer vision algorithm. | Methodology | `sensing`, `remembering`| 3 | 4 | 4 |
| **SURF (Speeded Up Robust Features)**| A faster approximation of SIFT. | Methodology | `sensing`, `remembering`| 3 | 4 | 4 |
| **ORB (Oriented FAST and Rotated BRIEF)**| A very fast and efficient binary descriptor, excellent for real-time applications. A good biomimetic choice for efficiency. | Methodology | `sensing`, `remembering`| 4 | 3 | 5 |

---

#### **Stage 2: SENSORY_PATTERNS - Intra-Visual Pattern Discovery**

**Goal:** To analyze the foundational features from Stage 1 to discover internal patterns, object groupings, and the overall composition of the visual scene. Populates `SENSORY_PATTERNS['individual_sense_patterns']['visual_patterns']`.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Blob Detection** | Identifies bright or dark regions on a contrasting background. Useful for finding general areas of interest. | Function/Formula | `sensing` | 5 | 2 | 5 |
| **Image Segmentation (Watershed, Felzenszwalb)**| Partitions an image into multiple segments or "superpixels" based on color and texture similarity. A key step in object recognition. | Function/Formula | `sensing`, `conceptualising`| 4 | 3 | 4 |
| **Contour Detection** | Finds the boundaries of shapes in the image after thresholding or edge detection. | Function/Formula | `sensing` | 5 | 2 | 5 |
| **Template Matching** | A "sliding window" search that looks for a small template image within a larger image. | Function/Formula | `remembering` | 4 | 2 | 4 |
| **Saliency Mapping (Itti-Koch-Niebur)**| A purely computational model that predicts which parts of an image a human would focus on based on low-level features. | Methodology | `sensing` | **5** | 4 | 4 |
| **Bag of Visual Words (BoVW)**| A classic method that involves clustering feature descriptors (like SIFT) to form a "visual vocabulary" and then representing the image as a histogram of these words. | Methodology | `conceptualising` | 3 | 4 | 3 |
| **Object Detection (YOLO, SSD - as tools)**| Uses a pre-trained, efficient model as a tool to get bounding boxes and class labels for common objects. | Model (Tool) | `sensing`, `reading` | 2 | 3 | 3 |

---

#### **Stage 3: FRAGMENT - Consolidation and Coherent Scene Generation**

**Goal:** To synthesize the visual features and patterns into a unified, coherent representation within the multi-modal `FRAGMENT`, generating descriptive summaries and preparing for deep analysis.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Graph-Based Synthesis (GNN)**| **Core Methodology:** Constructs a "scene graph" where detected objects/regions are nodes and their spatial relationships (e.g., "above," "next to") are edges. A GNN learns a holistic vector representation of the entire visual scene. | Methodology | `conceptualising`, `resolving`| 4 | 4 | 4 |
| **Image Captioning Models (as Tools)**| Uses a pre-trained model (e.g., a small Vision-Transformer based model like BLIP) as a utility to generate the `scene_description`. | Model (Tool) | `writing`, `thinking` | 2 | 3 | 3 |
| **Aesthetic Quality Assessment**| A model trained to predict human ratings of photographic quality. Used to populate the `aesthetic_score`. | Model (Tool) | `feeling` | 2 | 3 | 4 |
| **Information-Theoretic Scoring**| Applies functions like Lempel-Ziv Complexity to the image data to calculate the `visual_complexity` score objectively. | Function/Formula | `thinking` | 5 | 3 | 4 |

---

#### **Stage 4: NODE - Advanced Visual Reasoning and Network Integration**

**Goal:** "Conscious" analysis of the visual fragment. This involves deep contextual understanding, validating the scene against world knowledge, and integrating it into the brain's network.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I-JEPA (Image Joint-Embedding Predictive Architecture)**| **Primary World Model:** The ideal architecture for this stage. It takes the image fragment and processes it through its context encoder, predicting masked-out parts in an abstract representation space. The prediction error is a powerful measure of how well the image conforms to the AI's learned world model. High success = high `consensus_score`. | Methodology / Arch.| `learning`, `reasoning`, `sensing` | **5** | 4 | 2 |
| **Vision Transformer (ViT)** | **Feature Extractor:** While I-JEPA is the methodology, a ViT (without the final classifier) is often the core model used as the encoder. It excels at capturing global context and relationships between different parts of the image. | Model / Arch. | `thinking` | 3 | 4 | 3 |
| **Graph Neural Networks (GNNs)**| **Relational Reasoner:** Performs `node-to-node matching`. It takes the scene graph from the new node and compares it against the entire brain's knowledge graph to find connections based on object co-occurrence, spatial composition, and semantic similarity. | Model / Arch. | `reasoning` | 4 | 4 | 4 |
| **Coherence Validators** | **Dissonance Detector:** A suite of models/functions. For example, it could use a GNN to detect "impossible" spatial relationships (e.g., a "boat" node that is "inside" a "desert" node) or use a VQA (Visual Question Answering) model to probe the image for logical consistency. | Methodology | `reasoning`, `resolving`| 5 | 4 | 4 |
| **Diffusion Models (as Tools)**| **Generative Reasoner:** Can be used for advanced reasoning via "inpainting." By masking a region of the image and asking the model to fill it in, you can test how well the object fits the context. If the model generates something similar, the scene is coherent. | Model (Tool) | `conceptualising`| 3 | 4 | 2 |

---

#### **Stage 5: SEMANTIC_WORLD_MAP - Efficient Indexing for Retrieval**

**Goal:** To create a fast, searchable index of the validated node's visual content for quick retrieval.

| Name | Description | Type | Cognitive Processes | Biomimetic Fit | Complexity | Compute |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image Embedding Models (CLIP, DINO, SimCLR)**| **Semantic Indexer:** A pre-trained model used as a tool to create a single, dense vector embedding from the node's image. This is the key to semantic visual search ("find nodes that look like this"). | Model (Tool) | `remembering`, `searching` | 3 | 3 | 3 |
| **Tag/Keyword Index (BM25)** | **Keyword Indexer:** The descriptive tags, OCR'd text, and generated captions from the node are put into a classical sparse vector index for fast keyword-based retrieval. | Function/Formula | `remembering`, `searching` | 2 | 2 | 5 |
| **HNSW Index (FAISS, etc.)**| **Vector Database Engine:** The data structure that stores the image embeddings and allows for extremely fast similarity search. The core of "query-by-image." | Function/Formula | `remembering`, `searching` | 3 | 4 | 4 |
| **Perceptual Hashing (pHash, aHash, dHash)**| **Near-Duplicate Detector:** Creates a short, compact "hash" of an image. Images with similar hashes are visually almost identical. Excellent for finding duplicate or near-duplicate visual memories. | Function/Formula | `remembering`, `searching` | 4 | 2 | 5 |