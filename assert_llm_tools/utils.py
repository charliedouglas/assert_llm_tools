from nltk.corpus import stopwords
from typing import Set, Optional, List, Dict, Tuple, Union
import nltk
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

_custom_stopwords: Set[str] = set()


def initialize_nltk():
    """Initialize required NLTK data."""
    required_packages = [
        "punkt",
        "stopwords",
        "averaged_perceptron_tagger",
        "punkt_tab",
    ]

    for package in required_packages:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                # Some packages might have different paths
                nltk.download(package.replace("_tab", ""), quiet=True)


def add_custom_stopwords(words: List[str]) -> None:
    """
    Add custom words to the stopwords list.

    Args:
        words (List[str]): List of words to add to stopwords
    """
    global _custom_stopwords
    _custom_stopwords.update(set(word.lower() for word in words))


def get_all_stopwords() -> Set[str]:
    """
    Get combined set of NLTK and custom stopwords.

    Returns:
        Set[str]: Combined set of stopwords
    """
    return set(stopwords.words("english")).union(_custom_stopwords)


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning and normalizing it.
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Convert to lowercase
    text = text.lower()
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text using NLTK's stopwords list and custom stopwords.

    Args:
        text (str): Input text to process

    Returns:
        str: Text with stopwords removed
    """
    stop_words = get_all_stopwords()
    return " ".join([word for word in text.split() if word not in stop_words])


def initialize_pii_engines():
    """Initialize and return the PII analyzer and anonymizer engines."""
    
    try:
        # Import here to avoid dependencies if PII detection is not used
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
        from presidio_anonymizer import AnonymizerEngine
        
        # Import spacy - Presidio will handle the model loading
        import spacy
        
        # Make sure at least one spaCy model is installed
        if not spacy.util.is_package("en_core_web_sm"):
            try:
                logger.info("Downloading spaCy model en_core_web_sm...")
                spacy.cli.download("en_core_web_sm")
                logger.info("Download complete.")
            except Exception as e:
                logger.warning(f"Failed to download spaCy model: {e}")
                logger.warning("PII detection may not work properly without a spaCy model.")
        
        # Use default analyzer with built-in recognizers
        try:
            # Let Presidio handle everything - no custom patterns
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
            return analyzer, anonymizer
        except Exception as e:
            logger.error(f"Failed to initialize Presidio engines: {e}")
            return None, None
    
    except ImportError as e:
        logger.error(f"Failed to initialize PII engines: {e}")
        return None, None


def detect_and_mask_pii(
    text: str,
    entity_types: Optional[List[str]] = None,
    mask_char: str = "*",
    preserve_partial: bool = False,
) -> Tuple[str, Dict[str, List[Dict[str, Union[str, int, float]]]]]:
    """
    Detect and mask PII in text.
    
    Args:
        text: The text to scan for PII
        entity_types: List of entity types to detect (defaults to all supported types)
        mask_char: Character to use for masking (default: "*")
        preserve_partial: Whether to preserve part of the PII (e.g., for phone numbers: 123-***-***)
    
    Returns:
        Tuple containing:
            - The text with PII masked
            - A dictionary of detected entities with their original values and masked versions
    """
    if not text:
        return text, {}
    
    # Initialize engines
    analyzer, anonymizer = initialize_pii_engines()
    if not analyzer or not anonymizer:
        logger.warning("PII engines not initialized properly. Returning original text.")
        return text, {}
    
    try:
        # Import here to avoid dependencies if PII detection is not used
        from presidio_anonymizer.entities import OperatorConfig
        
        # Default entity types if none provided
        if not entity_types:
            entity_types = [
                "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", 
                "US_SSN", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "LOCATION",
                "NRP", "US_PASSPORT", "IP_ADDRESS", "DATE_TIME", "US_ITIN",
                "ORGANIZATION"
            ]
        
        # Analyze text for PII
        results = analyzer.analyze(text=text, entities=entity_types, language="en")
        
        # Build a mapping of detected PIIs
        detected_entities = {}
        
        # Check if any entities were found
        if not results:
            return text, detected_entities
        
        # Simple operator configuration
        if preserve_partial:
            # For partial masking, use entity-specific replacements
            operators = {
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": f"****@example.com"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": f"555-***-****"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": f"****-****-****-1234"}),
                "PERSON": OperatorConfig("replace", {"new_value": f"[Person]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": f"[Location]"}),
                "DATE_TIME": OperatorConfig("replace", {"new_value": f"[Date]"}),
                # Default is to use a generic replacement
                "DEFAULT": OperatorConfig("replace", {"new_value": f"[Partial {mask_char}{mask_char}{mask_char}]"})
            }
        else:
            # Use the built-in full masking operator with custom character
            operators = {
                "DEFAULT": OperatorConfig("replace", {"new_value": f"[{mask_char}{mask_char}{mask_char}{mask_char}{mask_char}]"})
            }
        
        # Anonymize text
        anonymized_result = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        
        # Create entity mapping for reference
        for item in results:
            entity_type = item.entity_type
            original_text = text[item.start:item.end]
            
            if entity_type not in detected_entities:
                detected_entities[entity_type] = []
            
            detected_entities[entity_type].append({
                "original": original_text,
                "start": item.start,
                "end": item.end,
                "score": item.score
            })
        
        return anonymized_result.text, detected_entities
        
    except Exception as e:
        logger.error(f"Error in PII detection/masking: {e}")
        return text, {}
