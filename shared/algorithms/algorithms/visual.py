import pandas as pd

# VISUAL SENSE - COMPLETE ALGORITHM CATALOG
visual_algorithms = {
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

# COLOR ANALYSIS ALGORITHMS
algorithms_data = [
    # Color Analysis
    ['RGB_to_HSV_Conversion', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Converts RGB color values to Hue-Saturation-Value color space for perceptual color analysis',
     'H=arctan2(√3*(G-B), 2*R-G-B), S=1-min(R,G,B)/V, V=max(R,G,B)',
     'RGB pixel values, resolution, bit_depth',
     'hue (0-360°), saturation (0.0-1.0), brightness (0.0-1.0)',
     'O(n)', 4, 'Use OpenCV cv2.cvtColor(). Essential for perceptual color analysis.'],
     
    ['Dominant_Color_Extraction_KMeans', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Uses K-means clustering on pixel colors to identify the N most representative colors in image',
     'Lloyd algorithm: minimize Σ||xi - μj||² over k clusters',
     'RGB pixel values, desired_cluster_count, color_space',
     'color_palette (top N colors), cluster_centers, color_percentages',
     'O(n*k*i)', 3, 'Use sklearn KMeans on reshaped pixel array. Choose k=3-8 for most images.'],
     
    ['Color_Histogram_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Computes frequency distribution of colors in RGB channels, creates color histograms',
     'Frequency counting: H(c) = count(pixels with color c) / total_pixels',
     'RGB pixel values, histogram_bins, image_format',
     'color_distribution histogram, brightness (0.0-1.0), contrast (0.0-1.0)',
     'O(n)', 4, 'Use numpy histogram. Normalize by total pixels. Standard bins=256.'],
     
    ['Color_Moments_Calculation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Calculates first 3 statistical moments (mean, variance, skewness) for each color channel',
     'μ = Σxi/n, σ² = Σ(xi-μ)²/n, skewness = Σ(xi-μ)³/(n*σ³)',
     'RGB pixel values per channel, resolution',
     'mean_colors, color_variance, color_skewness per RGB channel',
     'O(n)', 4, 'Use scipy.stats.moment(). Calculate per channel separately.'],
     
    ['Color_Correlogram', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Measures spatial correlation of color pairs - how colors are distributed relative to each other',
     'Correlation matrix: γ(d) = P(pixel at distance d has color pair (i,j))',
     'RGB pixel values, spatial coordinates (x,y), distance_parameter',
     'spatial_color_correlation_matrix, color_texture_measure',
     'O(n*d)', 3, 'Compute for multiple distance values. Expensive but very informative.'],
     
    ['White_Balance_Estimation', 'Function', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Estimates the color temperature of illumination to correct white balance automatically',
     'Gray world assumption: average color should be neutral gray',
     'RGB pixel values, camera_settings (if EXIF available)',
     'estimated_illuminant, white_balance_correction_matrix',
     'O(n)', 4, 'Use Gray World or White Patch algorithms. Check EXIF for camera WB.'],
     
    ['Color_Temperature_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Analyzes warmth/coolness of ambient light by examining color distribution bias',
     'Planckian locus fitting to color distribution in chromaticity space',
     'RGB pixel values, color_distribution',
     'color_temperature (K), warmth_index (-1.0 to 1.0)',
     'O(n)', 4, 'Map to chromaticity, fit to blackbody curve. 2700K=warm, 6500K=cool.'],
     
    ['Color_Constancy_Algorithm', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Color_Analysis',
     'Compensates for varying illumination conditions to maintain consistent color appearance',
     'Von Kries chromatic adaptation transform with illuminant estimation',
     'RGB pixel values, scene illumination estimate',
     'color_constancy_index, illumination_invariant_colors',
     'O(n)', 5, 'Biomimetic - mimics human visual adaptation. Use Retinex algorithm.'],

    # EDGE AND SHAPE DETECTION ALGORITHMS
    ['Sobel_Edge_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Computes gradient magnitude and direction using Sobel operators to detect edges',
     'Gx = [-1,0,1; -2,0,2; -1,0,1] * I, Gy = Gx.T, |G| = √(Gx² + Gy²)',
     'grayscale image, edge_threshold',
     'edge_strength, line_directions, gradient_magnitude, gradient_direction',
     'O(n)', 5, 'Fast edge detection. Use cv2.Sobel(). Biomimetic - similar to V1 cortex.'],
     
    ['Canny_Edge_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Multi-stage edge detection: gradient computation, non-maximum suppression, hysteresis',
     '∇I = [Gx,Gy], suppress non-maximum, threshold with hysteresis',
     'grayscale image, low_threshold, high_threshold',
     'edge_strength, contour_data, edge_continuity',
     'O(n)', 5, 'Gold standard edge detection. Use cv2.Canny(). Very biomimetic.'],
     
    ['Laplacian_of_Gaussian', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Applies Laplacian operator after Gaussian smoothing to find zero-crossings (edges)',
     '∇²G = ∂²G/∂x² + ∂²G/∂y², convolved with image, find zero crossings',
     'grayscale image, gaussian_sigma, zero_crossing_threshold',
     'edge_strength, blob_detection, zero_crossings',
     'O(n)', 4, 'Good for blob detection. Use cv2.Laplacian() after GaussianBlur.'],
     
    ['Hough_Line_Transform', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Detects straight lines by transforming edge points to Hough parameter space',
     'ρ = x*cos(θ) + y*sin(θ), accumulate votes in (ρ,θ) parameter space',
     'binary edge image, angle_resolution, distance_resolution',
     'line_types, line_directions, geometric_shapes (lines)',
     'O(n*θ)', 3, 'Use cv2.HoughLines(). Good for architectural/geometric images.'],
     
    ['Hough_Circle_Transform', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Detects circular objects by accumulating votes in 3D Hough space (x,y,radius)',
     '(x-a)² + (y-b)² = r², accumulate votes in (a,b,r) parameter space',
     'binary edge image, min_radius, max_radius, circle_threshold',
     'geometric_shapes (circles), object_positions, object_count',
     'O(n*r)', 3, 'Use cv2.HoughCircles(). Memory intensive but effective for circles.'],
     
    ['Harris_Corner_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Identifies corner points using second-moment matrix eigenvalue analysis',
     'M = [Ix² IxIy; IxIy Iy²], R = det(M) - k*trace(M)², find local maxima',
     'grayscale image, corner_threshold, harris_k_parameter',
     'corner_points, feature_points, geometric_shapes (corners)',
     'O(n)', 4, 'Use cv2.cornerHarris(). Essential for feature matching.'],
     
    ['SIFT_Feature_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Scale-invariant feature detection using Difference of Gaussians pyramid',
     'D(x,σ) = (G(x,kσ) - G(x,σ)) * I(x), find extrema across scale space',
     'grayscale image, octave_layers, contrast_threshold',
     'keypoints, descriptors, scale_invariant_features',
     'O(n*log(n))', 3, 'Use cv2.SIFT(). Patent issues - consider ORB alternative.'],
     
    ['Contour_Extraction', 'Function', 'SENSORY_RAW->PATTERNS', 'Edge_Shape_Detection',
     'Traces object boundaries to extract closed contours from binary edge images',
     'Chain code following algorithm on binary edge image',
     'binary edge image, contour_approximation_method',
     'contour_data, object_boundaries, shape_descriptors',
     'O(n)', 4, 'Use cv2.findContours(). Essential for shape analysis.'],

    # TEXTURE ANALYSIS ALGORITHMS
    ['Local_Binary_Patterns', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Texture_Analysis',
     'Describes local texture around each pixel by comparing with neighboring pixels',
     'LBP(x,y) = Σ s(gp - gc) * 2^p, where s(x) = 1 if x≥0, else 0',
     'grayscale image, radius, neighbor_points',
     'texture_types, texture_density, pattern_regularity, local_texture_map',
     'O(n)', 5, 'Use skimage.feature.local_binary_pattern. Highly biomimetic texture analysis.'],
     
    ['Gray_Level_Co_occurrence_Matrix', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Texture_Analysis',
     'Computes spatial relationship matrix of gray levels at specified distances/angles',
     'GLCM(i,j) = count of pixel pairs with values (i,j) at distance d, angle θ',
     'grayscale image, distances, angles, gray_levels',
     'texture_properties: contrast, correlation, energy, homogeneity',
     'O(n*d*θ)', 3, 'Use skimage.feature.greycomatrix(). Classic texture analysis method.'],
     
    ['Gabor_Filter_Bank', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Texture_Analysis',
     'Applies bank of Gabor filters at multiple orientations and frequencies for texture',
     'G(x,y) = exp(-(x²+γ²y²)/2σ²) * cos(2π(x cosθ + y sinθ)/λ + ψ)',
     'grayscale image, filter_frequencies, filter_orientations',
     'texture_orientation, texture_anisotropy, directional_texture_response',
     'O(n*f*θ)', 5, 'Very biomimetic - models V1 simple cells. Use cv2.getGaborKernel().'],
     
    ['Wavelet_Texture_Analysis', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Texture_Analysis',
     'Multi-resolution texture analysis using 2D wavelet decomposition',
     '2D DWT: W(j,k) = Σ f(x,y) * ψ(2^-j x - k, 2^-j y - l)',
     'grayscale image, wavelet_type, decomposition_levels',
     'texture_scale, multi_resolution_texture, texture_energy_distribution',
     'O(n*log(n))', 4, 'Use PyWavelets. Good for multi-scale texture analysis.'],
     
    ['Tamura_Texture_Features', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Texture_Analysis',
     'Computes 6 texture features: coarseness, contrast, directionality, line-likeness, regularity, roughness',
     'Complex formulas for each feature based on local neighborhood statistics',
     'grayscale image, neighborhood_size',
     'texture_coarseness, texture_contrast, texture_directionality, texture_regularity',
     'O(n)', 4, 'Perceptually meaningful texture features. Custom implementation needed.'],
     
    ['Laws_Texture_Energy', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Texture_Analysis',
     'Applies Laws texture masks to compute local texture energy measures',
     'Convolution with Laws masks: L5=[1,4,6,4,1], E5=[-1,-2,0,2,1], etc.',
     'grayscale image, laws_mask_set',
     'texture_energy_measures, texture_classification, surface_roughness',
     'O(n)', 3, 'Classic texture analysis. Use predefined 5x5 Laws masks.'],

    # OBJECT DETECTION ALGORITHMS  
    ['Blob_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Object_Detection',
     'Detects blob-like regions using Laplacian of Gaussian at multiple scales',
     'Scale-space blob detection using LoG extrema detection',
     'grayscale image, min_sigma, max_sigma, threshold',
     'blob_coordinates, blob_scales, objects_detected',
     'O(n*s)', 4, 'Use skimage.feature.blob_log. Good for round objects.'],
     
    ['Template_Matching', 'Function', 'SENSORY_RAW->PATTERNS', 'Object_Detection',
     'Matches predefined templates against image using normalized cross-correlation',
     'NCC(x,y) = Σ[I(x,y) - Ī][T(x,y) - T̄] / √(Σ[I-Ī]² * Σ[T-T̄]²)',
     'grayscale image, template_images, matching_threshold',
     'objects_detected, object_positions, object_confidence_scores',
     'O(n*m)', 3, 'Use cv2.matchTemplate(). Simple but effective for known objects.'],
     
    ['Cascade_Classifier', 'Model', 'SENSORY_RAW->PATTERNS', 'Object_Detection',
     'Uses Haar cascades or LBP cascades to detect specific object classes',
     'Boosted cascade of weak classifiers using Haar or LBP features',
     'grayscale image, cascade_file, scale_factor, min_neighbors',
     'faces_detected, object_positions, detection_confidence',
     'O(n)', 4, 'Use cv2.CascadeClassifier(). Pre-trained models available.'],
     
    ['Watershed_Segmentation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Object_Detection',
     'Segments image into regions using watershed algorithm for object separation',
     'Morphological watershed: treat image as topographic surface',
     'grayscale image, markers, connectivity',
     'segmented_regions, object_boundaries, separated_objects',
     'O(n*log(n))', 3, 'Use cv2.watershed(). Good for separating touching objects.'],

    # QUALITY ANALYSIS ALGORITHMS
    ['Sharpness_Assessment', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Measures image sharpness using gradient magnitude variance',
     'Sharpness = variance(gradient_magnitude) = var(√(Gx² + Gy²))',
     'grayscale image, gradient_method',
     'sharpness (0.0-1.0), focus_quality, blur_detection',
     'O(n)', 4, 'High variance = sharp. Use Sobel gradients for calculation.'],
     
    ['Noise_Level_Estimation', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Estimates noise level in image using wavelet-based method',
     'σ_noise = median(|wavelet_coefficients|) / 0.6745',
     'grayscale image, wavelet_type',
     'noise_level (0.0-1.0), noise_type, signal_to_noise_ratio',
     'O(n)', 3, 'Use wavelet decomposition. Robust noise estimation method.'],
     
    ['Blur_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Detects motion blur and focus blur using frequency domain analysis',
     'FFT analysis: blur creates characteristic frequency patterns',
     'grayscale image, blur_threshold',
     'blur_detection (0.0-1.0), blur_type, motion_blur_direction',
     'O(n*log(n))', 3, 'Use FFT analysis. Motion blur shows directional patterns.'],
     
    ['Compression_Artifacts_Detection', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Detects JPEG compression artifacts using block DCT analysis',
     'Analyze 8x8 block patterns characteristic of JPEG compression',
     'image (any format), compression_threshold',
     'compression_artifacts (0.0-1.0), compression_quality_estimate',
     'O(n)', 3, 'Look for 8x8 blocking patterns. Check DCT coefficients.'],
     
    ['Overall_Quality_Score', 'Function', 'SENSORY_RAW->PATTERNS', 'Quality_Analysis',
     'Computes composite quality metric combining multiple quality measures',
     'Weighted average: Q = w1*sharpness + w2*(1-noise) + w3*(1-blur) + w4*(1-compression)',
     'sharpness, noise_level, blur_detection, compression_artifacts',
     'overall_quality_score (0.0-1.0), quality_factors',
     'O(1)', 4, 'Combine individual quality metrics with learned weights.'],

    # DEPTH/3D ANALYSIS ALGORITHMS
    ['Depth_from_Focus', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Depth_Analysis',
     'Estimates depth using focus/defocus analysis across image regions',
     'Analyze local sharpness variations to infer depth',
     'image_stack (multiple focus levels) OR single image with focus analysis',
     'depth_map, foreground_elements, background_elements, depth_layers',
     'O(n*f)', 3, 'Requires multiple images or sophisticated focus analysis.'],
     
    ['Shape_from_Shading', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Depth_Analysis',
     'Recovers 3D shape from intensity variations assuming known lighting',
     'Reflectance equation: I(x,y) = ρ(x,y) * N(x,y) · L',
     'grayscale image, lighting_direction, surface_properties',
     'depth_cues, surface_normals, 3d_shape_estimate',
     'O(n)', 3, 'Requires assumptions about lighting and surface properties.'],
     
    ['Occlusion_Analysis', 'Function', 'SENSORY_RAW->PATTERNS', 'Depth_Analysis',
     'Analyzes which objects occlude others to determine depth ordering',
     'T-junction detection and boundary analysis for occlusion relationships',
     'segmented objects, object_boundaries',
     'occlusion_relationships, depth_ordering, layered_representation',
     'O(n²)', 4, 'Analyze object boundaries for T-junctions and occlusion cues.'],

    # LIGHTING ANALYSIS ALGORITHMS
    ['Shadow_Detection', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Lighting_Analysis',
     'Detects shadow regions using color and intensity analysis',
     'Analyze chromaticity changes and intensity drops characteristic of shadows',
     'RGB image, shadow_threshold, chromaticity_analysis',
     'shadows_presence, shadow_direction, lighting_type',
     'O(n)', 4, 'Use HSV color space. Shadows change V but preserve H.'],
     
    ['Light_Source_Direction', 'Algorithm', 'SENSORY_RAW->PATTERNS', 'Lighting_Analysis',
     'Estimates primary light source direction from shadow and shading patterns',
     'Analyze shadow directions and shading gradients',
     'RGB image, detected_shadows, surface_normals',
     'lighting_direction, light_intensity, lighting_quality',
     'O(n)', 4, 'Combine shadow analysis with shape-from-shading.'],
     
    ['Highlight_Detection', 'Function', 'SENSORY_RAW->PATTERNS', 'Lighting_Analysis',
     'Detects specular highlights and reflective regions',
     'Find regions with high intensity and color saturation loss',
     'RGB image, highlight_threshold',
     'highlights_presence, reflections_presence, surface_material_estimate',
     'O(n)', 3, 'Look for high intensity + low saturation regions.'],

    # =============================================================================
    # STAGE 2: PATTERNS -> FRAGMENTS  
    # =============================================================================
    
    ['Multi_Scale_Pattern_Integration', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Combines patterns detected at different scales into coherent multi-scale representation',
     'Hierarchical integration of scale-space features using weighted combination',
     'patterns from multiple scales, scale_weights, integration_method',
     'integrated_pattern_hierarchy, multi_scale_coherence_score',
     'O(n*s)', 5, 'Combine results from different scale analyses. Very biomimetic approach.'],
     
    ['Spatial_Pattern_Clustering', 'Algorithm', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Groups spatially proximate patterns into coherent clusters',
     'DBSCAN clustering on spatial coordinates of detected patterns',
     'pattern_locations, pattern_types, clustering_parameters',
     'pattern_clusters, spatial_relationships, clustered_patterns',
     'O(n²)', 4, 'Use DBSCAN with spatial distance metric.'],
     
    ['Color_Shape_Association', 'Function', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Associates detected colors with detected shapes for object recognition',
     'Spatial overlap analysis between color regions and shape regions',
     'color_regions, shape_regions, spatial_coordinates',
     'color_shape_associations, object_hypotheses',
     'O(n*m)', 4, 'Calculate spatial overlap between color and shape detections.'],
     
    ['Texture_Color_Correlation', 'Function', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Correlates texture patterns with color patterns for material recognition',
     'Statistical correlation between texture features and color features',
     'texture_patterns, color_patterns, spatial_alignment',
     'material_hypotheses, texture_color_correlations',
     'O(n)', 3, 'Use Pearson correlation between texture and color features.'],
     
    ['Object_Scene_Context_Analysis', 'Methodology', 'PATTERNS->FRAGMENTS', 'Pattern_Integration',
     'Analyzes detected objects within scene context for coherence validation',
     'Contextual relationship analysis using co-occurrence statistics',
     'detected_objects, scene_classification, spatial_relationships',
     'context_coherence, object_scene_consistency, contextual_validation',
     'O(n²)', 4, 'Check if detected objects make sense together in scene context.'],

    # =============================================================================
    # STAGE 3: FRAGMENTS -> NODES
    # =============================================================================
    
    ['Visual_Concept_Formation', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Forms abstract visual concepts from integrated visual fragments',
     'Hierarchical clustering and abstraction of visual patterns',
     'integrated_fragments, concept_hierarchy_rules, abstraction_level',
     'visual_concepts, concept_hierarchy, concept_confidence',
     'O(n²)', 5, 'Create abstract visual concepts. Highly biomimetic process.'],
     
    ['Object_Recognition_Integration', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Integrates multiple visual cues into coherent object recognition',
     'Weighted evidence combination from multiple visual modalities',
     'shape_evidence, color_evidence, texture_evidence, context_evidence',
     'object_recognition_results, recognition_confidence, object_categories',
     'O(n)', 4, 'Combine multiple types of visual evidence for object recognition.'],
     
    ['Scene_Understanding', 'Methodology', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Develops high-level understanding of complete visual scene',
     'Graph-based scene representation with objects and relationships',
     'recognized_objects, spatial_relationships, scene_context',
     'scene_graph, scene_description, scene_categories',
     'O(n²)', 4, 'Build complete scene understanding from visual components.'],
     
    ['Visual_Memory_Encoding', 'Function', 'FRAGMENTS->NODES', 'Concept_Formation',
     'Encodes visual information for long-term storage and retrieval',
     'Hierarchical encoding with compression and key feature preservation',
     'visual_concepts, importance_weights, encoding_strategy',
     'encoded_visual_memory, memory_keys, retrieval_cues',
     'O(n)', 4, 'Compress visual information while preserving important features.'],

    # =============================================================================
    # STAGE 4: NODES -> SEMANTIC_WORLD_MAP
    # =============================================================================
    
    ['Visual_Embedding_Generation', 'Model', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Generates dense vector embeddings for visual content using pre-trained models',
     'Deep CNN feature extraction: ResNet, CLIP, or similar architectures',
     'visual_nodes, pre_trained_model, embedding_dimensions',
     'visual_embeddings, embedding_vectors, similarity_indices',
     'O(n)', 3, 'Use CLIP or ResNet features. Transform to fixed-size vectors.'],
     
    ['Visual_Similarity_Index', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates searchable index for visual similarity using embeddings',
     'HNSW (Hierarchical Navigable Small World) approximate nearest neighbor',
     'visual_embeddings, index_parameters, similarity_threshold',
     'similarity_index, nearest_neighbor_structure, search_capability',
     'O(n*log(n))', 3, 'Use FAISS library for efficient similarity search.'],
     
    ['Visual_Tag_Generation', 'Function', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Generates searchable tags and keywords for visual content',
     'Extract descriptive keywords from visual concepts and scene understanding',
     'visual_concepts, scene_descriptions, object_categories',
     'visual_tags, keyword_index, searchable_descriptors',
     'O(n)', 3, 'Generate text tags from visual analysis for keyword search.'],
     
    ['Perceptual_Hash_Generation', 'Algorithm', 'NODES->SEMANTIC_MAP', 'Indexing_Retrieval',
     'Creates perceptual hashes for duplicate and near-duplicate detection',
     'Average Hash, Perceptual Hash, or Difference Hash algorithms',
     'visual_content, hash_algorithm_type, hash_size',
     'perceptual_hash, duplicate_detection_capability, hash_distance',
     'O(1)', 4, 'Use imagehash library. Fast duplicate detection capability.']
]

# Add all the data to the dictionary
for algo_data in algorithms_data:
    visual_algorithms['Algorithm_Name'].append(algo_data[0])
    visual_algorithms['Type'].append(algo_data[1]) 
    visual_algorithms['Stage'].append(algo_data[2])
    visual_algorithms['Category'].append(algo_data[3])
    visual_algorithms['Description'].append(algo_data[4])
    visual_algorithms['Mathematical_Basis'].append(algo_data[5])
    visual_algorithms['Input_Data_Required'].append(algo_data[6])
    visual_algorithms['Output_Data_Generated'].append(algo_data[7])
    visual_algorithms['Computational_Complexity'].append(algo_data[8])
    visual_algorithms['Biomimetic_Relevance'].append(algo_data[9])
    visual_algorithms['Implementation_Notes'].append(algo_data[10])

# Create DataFrame
visual_df = pd.DataFrame(visual_algorithms)

# Display summary
print("VISUAL SENSE - COMPLETE ALGORITHM CATALOG")
print("=" * 50)
print(f"Total Visual Algorithms: {len(visual_df)}")
print(f"Stage 1 (SENSORY_RAW->PATTERNS): {len(visual_df[visual_df['Stage'] == 'SENSORY_RAW->PATTERNS'])}")
print(f"Stage 2 (PATTERNS->FRAGMENTS): {len(visual_df[visual_df['Stage'] == 'PATTERNS->FRAGMENTS'])}")
print(f"Stage 3 (FRAGMENTS->NODES): {len(visual_df[visual_df['Stage'] == 'FRAGMENTS->NODES'])}")
print(f"Stage 4 (NODES->SEMANTIC_MAP): {len(visual_df[visual_df['Stage'] == 'NODES->SEMANTIC_MAP'])}")
print()

# Show category breakdown
print("ALGORITHM CATEGORIES:")
category_counts = visual_df['Category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} algorithms")
print()

# Show type breakdown  
print("ALGORITHM TYPES:")
type_counts = visual_df['Type'].value_counts()
for algo_type, count in type_counts.items():
    print(f"  {algo_type}: {count} algorithms")
print()

# Show sample of the complete data
print("SAMPLE ALGORITHM DETAILS:")
print(visual_df[['Algorithm_Name', 'Type', 'Stage', 'Category', 'Description']].head(10).to_string(index=False))

# Save to Excel if needed
# visual_df.to_excel('visual_algorithms_complete_catalog.xlsx', index=False)