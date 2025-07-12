text_crossmodal = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "fk_audio_id": "UUID",
    "fk_visual_id": "UUID",
    "fk_glyph_id": "UUID",
    "fk_physics_id": "UUID",
    "fk_emotion_id": "UUID",
    "fk_metaphysics_id": "UUID",
}

text_fragment = {
    "id": "UUID",
    "raw_text": "str",
    "source_type": "enum['spoken', 'written', 'typed']",
    "input_device": "enum['mic', 'keyboard', 'ocr', 'handwriting']",
    "data_source": "enum['user_input', 'self_generated', 'academic_source', 'social_media', 'news_source', 'generally_available', 'business_research', 'book']",
    "register": "enum['formal', 'informal', 'technical', 'conversational', 'divine']",
    "text_type": "enum['narrative', 'poetry', 'prose', 'dialogue', 'instruction', 'lyrics', 'code', 'other']",
    "text_subtype": "enum['fiction', 'nonfiction', 'technical', 'creative', 'formal', 'informal', 'academic', 'conversational', 'poetic', 'dramatic', 'scientific', 'historical', 'biographical', 'journalistic', 'advertising', 'legal', 'medical', 'educational', 'motivational', 'inspirational', 'self-help', 'spiritual', 'religious', 'philosophical', 'existential', 'metaphysical', 'ontological', 'epistemological', 'axiological', 'ethical', 'political', 'social', 'economic', 'environmental', 'cultural', 'linguistic', 'literary', 'artistic', 'musical', 'cinematic', 'performative', 'interactive', 'multimodal', 'other']",
    "text_content": "str",
    "fragment_type": "enum['character', 'word', 'phrase', 'sentence', 'paragraph']",
    "word_roots": "dict[str, int]",
    "affixes": "dict[str, int]",
    "compound_structures": "list[str]",
    "ambiguous_terms": "list[str]",
    "implied_meanings": "list[str]",
    "stylistic_features": "list[str]",
    "language": "enum[english, afrikaans,french, dutch,zulu,xhosa,nso,tswana,swati,venda,tsonga,sotho,ndebele,tswana]",
}

text_emotional = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "emotional_tone": "str",
    "emotional_intensity": "float",
    "arousal_valence": "dict[str, float]",
    "emotional_associations": "list[str]",
    "emotional_context": "str",
}

text_lexical = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "word_count": "int",
    "char_count": "int",
    "sentence_length": "int",
    "avg_word_length": "float",
    "syllable_count": "int",
    "syllable_pattern": "str",
    "character_count": "int",
    "syllable_count": "int",
    "phoneme_count": "int",
    "vowel_ratio": "float",
    "consonant_ratio": "float",
    "is_palindrome": "bool",
    "contains_glyph": "bool"
}

text_metaphysical_label = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "archetype": "str",
    "symbolism": "str",
    "conceptual_depth": "float",
    "hidden_meaning": "str",
    "personal_meaning": "str",
    "associations": "list[str]",
    "resonance": "float",
}

text_phonetic = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "ipa_transcription": "str",
    "phoneme_sequence": "list[str]",
    "stress_pattern": "str",
    "intonation": "enum['neutral', 'rising', 'falling', 'mixed']",
    "tone_quality": "enum['nasal', 'breathy', 'clear', 'tense']"
}

text_semantic = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "meaning": "str",
    "summary": "str",
    "categories": "list[str]",  # e.g. ['noun', 'emotion', 'action']
    "sentiment": "float",  # range: -1 to 1
    "certainty_score": "float",  # 0 to 1
    "language_tags": "list[str]",  # e.g. ['past-tense', 'imperative']
    "named_entity": "str",
    "language_ontology_refs": "list[str]",  # references to language ontologies
    "synonyms": "list[str]",  # synonyms for the meaning
    "antonyms": "list[str]",  # antonyms for the meaning
    "lemmas": "list[str]",  # lemmas for the meaning
    "domain_matrix_data": "list[float]",  # data for domain matrix
}  

text_spatial_shape = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "vector_path_id": "UUID",
    "word_shape_type": "enum['round', 'angular', 'curved', 'zigzag']",
    "complexity_score": "float",
    "visual_complexity": "float",
    "spatial_coherence": "float",
    "temporal_coherence": "float",
}

text_syntax_labels = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "POS_Tags": "list[str]",
    "syntax_tree_JSON": "list[str]",
    "syntax_tree_language": "str",
    "syntax_tree_language_version": "str",
    "syntax_tree_type": "str",
    "syntax_tree_version": "str",
    "syntax_tree_source": "str",
    "syntax_tree_source_version": "str",
    "syntax_tree_source_url": "str",
}

text_temporal = {
    "id": "UUID",
    "fk_fragment_id": "UUID",
    "historical_era": "str",
    "decay_rate": "float",
}




