import pandas as pd

# Text Sense Complete Algorithm Catalog
text_algorithms = {
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

# STAGE 1: SENSORY_RAW -> PATTERNS

# Character Analysis Algorithms (NEW - addressing gaps)
text_algorithms['Algorithm_Name'].extend([
    'Character_Recognition_OCR',
    'Character_Frequency_Analysis',
    'Character_Shape_Analysis',
    'Character_Spacing_Analysis',
    'Letter_Combination_Analysis',
    'Character_Position_Mapping',
    'Handwriting_Stroke_Analysis',
    'Capital_Lowercase_Pattern_Detection'
])

text_algorithms['Type'].extend([
    'Model', 'Function', 'Algorithm', 'Function', 
    'Function', 'Function', 'Algorithm', 'Function'
])

text_algorithms['Stage'].extend(['SENSORY_RAW->PATTERNS'] * 8)
text_algorithms['Category'].extend(['Character_Analysis'] * 8)

text_algorithms['Description'].extend([
    'Identifies individual characters from image using optical character recognition neural networks',
    'Counts occurrence frequency of each character in text for statistical analysis',
    'Analyzes geometric properties of character shapes: height, width, curves, angles',
    'Measures spacing between characters (kerning) and between words for typography analysis',
    'Identifies common letter combinations (bigrams, trigrams) and their frequencies',
    'Maps X,Y coordinates of each character for spatial text layout analysis',
    'Analyzes pen stroke direction, order, and pressure for handwritten text (if available)',
    'Detects patterns in capitalization usage throughout text'
])

text_algorithms['Mathematical_Basis'].extend([
    'CNN architecture: Conv2D -> MaxPool -> Dense -> Softmax for character classification',
    'Frequency count: f(c) = count(character c) / total_characters',
    'Geometric feature extraction: aspect_ratio = height/width, curve_density = curves/perimeter',
    'Distance metrics: character_spacing = x[i+1] - (x[i] + width[i])',
    'N-gram analysis: P(c[i+1]|c[i]) = count(c[i]c[i+1]) / count(c[i])',
    'Coordinate extraction from bounding boxes: (x_min, y_min, x_max, y_max)',
    'Vector field analysis of stroke direction, curvature calculation: κ = |x\'y\'\' - y\'x\'\'| / (x\'² + y\'²)^(3/2)',
    'Pattern matching: regex patterns for capitalization rules'
])

text_algorithms['Input_Data_Required'].extend([
    'text_content (image), resolution, character_count',
    'text_content (string), character_count, encoding',
    'individual_characters (list), text_content (image)',
    'character_positions, individual_characters, font_metrics',
    'text_content (string), character_count, word_boundaries',
    'text_content (image), character_bounding_boxes',
    'text_content (image), stroke_data, pen_pressure (if available)',
    'text_content (string), word_boundaries, sentence_boundaries'
])

text_algorithms['Output_Data_Generated'].extend([
    'individual_characters (list), character_confidence_scores, character_positions',
    'character_frequencies (dict), character_diversity_index',
    'character_shapes (dict): height, width, area, perimeter, curve_count per character',
    'character_spacing (list), word_spacing (list), line_spacing (list)',
    'letter_combinations (dict), bigram_frequencies, trigram_frequencies',
    'character_positions (list of x,y coordinates), text_layout_structure',
    'character_stroke_data (dict), stroke_order, pen_pressure_variation',
    'capital_lowercase_patterns (dict), capitalization_rules_detected'
])

# Phonetic Analysis Algorithms (NEW - addressing gaps)
text_algorithms['Algorithm_Name'].extend([
    'Grapheme_to_Phoneme_Conversion',
    'Syllable_Boundary_Detection',
    'Phoneme_Frequency_Analysis',
    'Rhyme_Pattern_Detection',
    'Alliteration_Analysis',
    'Phonetic_Similarity_Calculation',
    'Pronunciation_Difficulty_Assessment',
    'IPA_Transcription_Generation'
])

text_algorithms['Type'].extend([
    'Model', 'Algorithm', 'Function', 'Algorithm',
    'Algorithm', 'Function', 'Function', 'Model'
])

text_algorithms['Stage'].extend(['SENSORY_RAW->PATTERNS'] * 8)
text_algorithms['Category'].extend(['Phonetic_Analysis'] * 8)

text_algorithms['Description'].extend([
    'Converts written text to phonemic representation using neural sequence-to-sequence models',
    'Identifies syllable boundaries in words using phonotactic constraints and stress patterns',
    'Analyzes frequency distribution of phonemes in text for phonological pattern recognition',
    'Detects rhyming patterns by comparing phoneme endings of words',
    'Identifies alliterative patterns by analyzing initial phonemes of adjacent words',
    'Calculates phonetic similarity between words using phoneme edit distance',
    'Assesses pronunciation complexity based on phoneme combinations and articulatory features',
    'Generates International Phonetic Alphabet transcription for accurate pronunciation'
])

text_algorithms['Mathematical_Basis'].extend([
    'Seq2Seq model: P(phoneme|grapheme) using attention mechanism',
    'Onset-nucleus-coda syllable structure, Maximum Onset Principle',
    'Frequency distribution: f(p) = count(phoneme p) / total_phonemes',
    'Phoneme suffix matching: similarity(w1,w2) = LCS(suffix(w1), suffix(w2))',
    'Initial phoneme clustering: group words by first phoneme',
    'Levenshtein distance on phoneme sequences: edit_distance(p1, p2)',
    'Articulatory complexity score: Σ feature_difficulty(phoneme)',
    'Rule-based + neural hybrid: context-dependent phoneme selection'
])

# Linguistic Analysis Algorithms
text_algorithms['Algorithm_Name'].extend([
    'Tokenization_BPE',
    'Part_of_Speech_Tagging',
    'Named_Entity_Recognition',
    'Dependency_Parsing',
    'Morphological_Analysis',
    'Word_Frequency_Analysis',
    'N_Gram_Generation',
    'Collocation_Detection',
    'Sentence_Boundary_Detection',
    'Language_Detection'
])

text_algorithms['Type'].extend([
    'Algorithm', 'Model', 'Model', 'Model', 'Algorithm',
    'Function', 'Function', 'Algorithm', 'Algorithm', 'Model'
])

text_algorithms['Stage'].extend(['SENSORY_RAW->PATTERNS'] * 10)
text_algorithms['Category'].extend(['Linguistic_Analysis'] * 10)

text_algorithms['Description'].extend([
    'Segments text into subword tokens using Byte-Pair Encoding for handling out-of-vocabulary words',
    'Assigns grammatical categories (noun, verb, adjective, etc.) to each token using neural tagger',
    'Identifies and classifies named entities (person, location, organization) in text',
    'Constructs syntactic dependency tree showing grammatical relationships between words',
    'Analyzes word structure to identify roots, prefixes, suffixes, and inflectional morphemes',
    'Computes frequency distribution of words following Zipf\'s law for vocabulary analysis',
    'Generates N-gram sequences (bigrams, trigrams) for language modeling and pattern detection',
    'Identifies statistically significant word combinations that occur together frequently',
    'Detects sentence boundaries using punctuation, capitalization, and contextual cues',
    'Identifies the language of text using character/word n-gram distributions'
])

# Semantic Analysis Algorithms
text_algorithms['Algorithm_Name'].extend([
    'Word2Vec_Skip_Gram',
    'TF_IDF_Vectorization',
    'Topic_Modeling_LDA',
    'Sentiment_Analysis_VADER',
    'Keyword_Extraction_RAKE',
    'Text_Similarity_Cosine',
    'Word_Sense_Disambiguation',
    'Coreference_Resolution',
    'Semantic_Role_Labeling'
])

text_algorithms['Type'].extend([
    'Model', 'Algorithm', 'Model', 'Function', 'Algorithm',
    'Function', 'Model', 'Model', 'Model'
])

text_algorithms['Stage'].extend(['SENSORY_RAW->PATTERNS'] * 9)
text_algorithms['Category'].extend(['Semantic_Analysis'] * 9)

# STAGE 2: PATTERNS -> FRAGMENTS
text_algorithms['Algorithm_Name'].extend([
    'Cross_Modal_Text_Integration',
    'Semantic_Pattern_Clustering',
    'Context_Coherence_Analysis',
    'Narrative_Structure_Detection',
    'Thematic_Analysis'
])

text_algorithms['Type'].extend(['Methodology', 'Algorithm', 'Function', 'Algorithm', 'Algorithm'])
text_algorithms['Stage'].extend(['PATTERNS->FRAGMENTS'] * 5)
text_algorithms['Category'].extend(['Pattern_Integration'] * 5)

# STAGE 3: FRAGMENTS -> NODES
text_algorithms['Algorithm_Name'].extend([
    'Semantic_Graph_Construction',
    'Concept_Hierarchy_Building',
    'Knowledge_Graph_Integration',
    'Contextual_Meaning_Resolution'
])

text_algorithms['Type'].extend(['Methodology', 'Algorithm', 'Methodology', 'Algorithm'])
text_algorithms['Stage'].extend(['FRAGMENTS->NODES'] * 4)
text_algorithms['Category'].extend(['Knowledge_Construction'] * 4)

# STAGE 4: NODES -> SEMANTIC_WORLD_MAP
text_algorithms['Algorithm_Name'].extend([
    'Text_Embedding_SBERT',
    'BM25_Keyword_Index',
    'Semantic_Search_FAISS',
    'Topic_Hierarchy_Index'
])

text_algorithms['Type'].extend(['Model', 'Algorithm', 'Function', 'Algorithm'])
text_algorithms['Stage'].extend(['NODES->SEMANTIC_MAP'] * 4)
text_algorithms['Category'].extend(['Indexing_Retrieval'] * 4)

# Fill remaining required data for all entries
total_algorithms = len(text_algorithms['Algorithm_Name'])

# Computational Complexity
text_algorithms['Computational_Complexity'].extend([
    'O(n)', 'O(n)', 'O(n²)', 'O(n)', 'O(n)', 'O(n)', 'O(n²)', 'O(n)',  # Character Analysis
    'O(n²)', 'O(n)', 'O(n)', 'O(n²)', 'O(n)', 'O(n²)', 'O(n)', 'O(n²)',  # Phonetic Analysis
    'O(n)', 'O(n²)', 'O(n²)', 'O(n³)', 'O(n)', 'O(n)', 'O(n)', 'O(n²)', 'O(n)', 'O(n)',  # Linguistic Analysis
    'O(n²)', 'O(n²)', 'O(n³)', 'O(n)', 'O(n)', 'O(n²)', 'O(n³)', 'O(n³)', 'O(n²)',  # Semantic Analysis
    'O(n²)', 'O(n²)', 'O(n)', 'O(n²)', 'O(n²)',  # Pattern Integration
    'O(n²)', 'O(n²)', 'O(n³)', 'O(n²)',  # Knowledge Construction
    'O(n²)', 'O(n)', 'O(log n)', 'O(n²)'  # Indexing/Retrieval
])

# Biomimetic Relevance (1-5 scale)
text_algorithms['Biomimetic_Relevance'].extend([
    5, 4, 3, 3, 4, 3, 2, 3,  # Character Analysis - high relevance for reading development
    5, 5, 4, 4, 4, 3, 3, 4,  # Phonetic Analysis - critical for speech development
    2, 3, 3, 4, 4, 4, 3, 3, 4, 2,  # Linguistic Analysis - mixed biomimetic relevance
    3, 2, 3, 2, 3, 3, 4, 4, 4,  # Semantic Analysis
    4, 4, 4, 4, 3,  # Pattern Integration
    4, 4, 3, 4,  # Knowledge Construction
    2, 2, 2, 3   # Indexing/Retrieval - more computational than biomimetic
])

# Implementation Notes
implementation_notes = [
    'Use Tesseract OCR or custom CNN. Requires image preprocessing.',
    'Simple counting algorithm. Unicode normalization recommended.',
    'Requires computer vision libraries (OpenCV). Extract contours first.',
    'Needs font metrics or OCR bounding boxes for accurate spacing.',
    'Use sliding window approach. Consider language-specific patterns.',
    'OCR preprocessing required. Store as coordinate arrays.',
    'Requires specialized hardware/software for stroke capture.',
    'Use regex patterns combined with linguistic rules.',
    
    'Use G2P neural models like Phonemizer or espeak-ng.',
    'Implement using phonotactic rules + ML classifier.',
    'Requires phoneme inventory for target language.',
    'Compare word endings using phoneme similarity metrics.',
    'Analyze initial sounds of consecutive words.',
    'Use Levenshtein distance on phoneme sequences.',
    'Based on articulatory feature complexity scores.',
    'Use IPA symbol set. Language-specific pronunciation rules.',
    
    'Use sentencepiece or Hugging Face tokenizers.',
    'Use spaCy or NLTK POS taggers. Pre-trained models available.',
    'Use spaCy NER or Hugging Face transformers.',
    'Use spaCy dependency parser or Stanford CoreNLP.',
    'Use NLTK morphological analyzers or custom rule sets.',
    'Simple counting with stopword filtering optional.',
    'Use NLTK n-gram generators. Configurable n values.',
    'Use statistical measures like PMI or log-likelihood ratio.',
    'Use NLTK Punkt tokenizer or spaCy sentence segmenter.',
    'Use langdetect library or custom n-gram classifiers.',
    
    'Use gensim Word2Vec implementation.',
    'Use sklearn TfidfVectorizer.',
    'Use gensim LDA implementation.',
    'Use VADER lexicon-based approach in NLTK.',
    'Use RAKE implementation or custom keyword extraction.',
    'Use sklearn cosine_similarity on TF-IDF vectors.',
    'Use NLTK WordNet or BabelNet for sense disambiguation.',
    'Use spaCy coreference resolution or neuralcoref.',
    'Use AllenNLP semantic role labeling models.',
    
    'Combine text patterns with other sensory modalities.',
    'Use clustering algorithms like K-means on semantic vectors.',
    'Analyze semantic consistency across text segments.',
    'Use discourse markers and rhetorical structure theory.',
    'Use topic modeling combined with sentiment analysis.',
    
    'Use NetworkX for graph construction and analysis.',
    'Build hierarchical structures using WordNet or ConceptNet.',
    'Integrate with existing knowledge bases like Wikidata.',
    'Use context vectors and disambiguation algorithms.',
    
    'Use sentence-transformers library for embeddings.',
    'Use Elasticsearch or custom BM25 implementation.',
    'Use Facebook FAISS for approximate nearest neighbor search.',
    'Build hierarchical topic structures using LDA hierarchy.'
]

text_algorithms['Implementation_Notes'].extend(implementation_notes)

# Create DataFrame
text_df = pd.DataFrame(text_algorithms)

print("Text Algorithms Catalog - Sample entries:")
print(text_df[['Algorithm_Name', 'Type', 'Stage', 'Category']].head(15).to_string(index=False))
print(f"\nTotal Text Algorithms: {len(text_df)}")
print(f"Character Analysis: {len([x for x in text_df['Category'] if x == 'Character_Analysis'])}")
print(f"Phonetic Analysis: {len([x for x in text_df['Category'] if x == 'Phonetic_Analysis'])}")