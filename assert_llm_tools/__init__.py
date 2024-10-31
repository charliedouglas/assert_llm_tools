import nltk
import os
from pathlib import Path


def initialize_nltk():
    """
    Initialize required NLTK data, downloading if necessary.
    Creates a data directory in the user's home directory if needed.
    """
    try:
        # Set NLTK data path to user's home directory
        nltk_data_dir = str(Path.home() / ".assert_tools" / "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        # List of required NLTK resources
        required_resources = ["punkt", "punkt_tab"]

        # Download all required resources
        for resource in required_resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, download_dir=nltk_data_dir, quiet=False)

        return True
    except Exception as e:
        print(f"Warning: Failed to initialize NLTK data: {str(e)}")
        return False


# Initialize when the package is imported
print("Initializing NLTK resources...")
initialize_nltk()
print("NLTK initialization complete.")
