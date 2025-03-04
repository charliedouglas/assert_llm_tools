from typing import Dict, Optional
import torch
from comet import download_model, load_from_checkpoint


class CustomCOMETScorer:
    def __init__(self, model_name: str = "wmt20-comet-da"):
        """Initialize COMET scorer with specified model.

        Args:
            model_name: COMET model to use. Default is 'wmt20-comet-da'
        """
        # Download and load the model (it will be cached after first download)
        try:
            print(f"Loading COMET model: {model_name}")
            model_path = download_model(model_name)
            self.model = load_from_checkpoint(model_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error loading COMET model: {e}")
            raise

    def score(self, source: str, reference: str, candidate: str) -> float:
        """Calculate COMET score manually without using the predict API.

        Args:
            source: The original full text
            reference: A reference summary
            candidate: The generated summary to evaluate

        Returns:
            float: COMET score (higher is better)
        """
        # Create a single sample
        sample = {"src": source, "mt": candidate, "ref": reference}

        try:
            # Manually encode the input
            with torch.no_grad():
                # Get model inputs (tokenize and encode)
                model_inputs = self.model.prepare_sample([sample])

                # Move inputs to the correct device
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(self.device)

                # Get model prediction
                model_output = self.model(**model_inputs)

                # Extract the score
                score = model_output["score"].cpu().item()

                return score
        except Exception as e:
            print(f"Error during COMET scoring: {e}")
            # Fallback score in case of error
            return 0.5


def calculate_comet_score(
    source: str, reference: str, candidate: str, model_name: str = "wmt20-comet-da"
) -> Dict[str, float]:
    """
    Calculate COMET score for a candidate summary against a reference and source.

    Args:
        source: The original full text
        reference: A reference summary
        candidate: The generated summary to evaluate
        model_name: COMET model to use

    Returns:
        Dict[str, float]: Dictionary containing the COMET score
    """
    try:
        # Initialize scorer
        scorer = CustomCOMETScorer(model_name=model_name)

        # Calculate score
        score = scorer.score(source, reference, candidate)

        return {"comet_score": score}
    except Exception as e:
        print(f"Failed to calculate COMET score: {e}")
        return {"comet_score": 0.0, "error": str(e)}


def calculate_comet_qe_score(
    source: str, candidate: str, model_name: str = "wmt20-comet-qe-da"
) -> Dict[str, float]:
    """
    Calculate COMET QE (Quality Estimation) score without reference summary.

    Args:
        source: The original full text
        candidate: The generated summary to evaluate
        model_name: COMET QE model to use

    Returns:
        Dict[str, float]: Dictionary containing the COMET QE score
    """
    try:
        # For QE models, we pass an empty reference
        scorer = CustomCOMETScorer(model_name=model_name)
        score = scorer.score(source, "", candidate)

        return {"comet_qe_score": score}
    except Exception as e:
        print(f"Failed to calculate COMET QE score: {e}")
        return {"comet_qe_score": 0.0, "error": str(e)}
