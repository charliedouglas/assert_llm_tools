import os
import json
import torch
from comet import download_model, load_from_checkpoint
from comet.models import CometModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset


def prepare_summarization_data(dataset_name="cnn_dailymail", split="train[:1000]"):
    """Prepare data for fine-tuning COMET on summarization.

    Args:
        dataset_name: HuggingFace dataset name
        split: Which split to use and how many examples

    Returns:
        List of examples in COMET format with human judgments
    """
    # Load dataset - in real scenario, you would need a dataset with human judgments
    dataset = load_dataset(dataset_name, "3.0.0", split=split)

    # This is a placeholder - in a real scenario, you would have human judgments
    # Here we're simulating them with a simple length-based heuristic
    examples = []
    for item in dataset:
        # Extract source and reference
        source = item["article"]
        reference = item["highlights"]

        # For demonstration: create a "bad" summary by truncating
        bad_summary = reference[: len(reference) // 3]

        # Simulate human scores (in practice, you need real human evaluations)
        # Higher score = better quality (0-1 range)
        good_score = 0.9  # Pretend reference is high quality
        bad_score = 0.3  # Pretend truncated summary is low quality

        # Add examples in COMET format
        examples.append(
            {
                "src": source,
                "mt": reference,  # Using reference as machine translation (good summary)
                "ref": reference,  # Same here for demonstration
                "score": good_score,
            }
        )

        examples.append(
            {
                "src": source,
                "mt": bad_summary,  # Bad/truncated summary
                "ref": reference,
                "score": bad_score,
            }
        )

    return examples


def finetune_comet_for_summarization(
    output_dir="./comet-summarization",
    base_model="wmt20-comet-da",
    batch_size=4,
    max_epochs=2,
    learning_rate=5e-6,
):
    """Fine-tune COMET for summarization evaluation.

    Args:
        output_dir: Directory to save the fine-tuned model
        base_model: Base COMET model to start from
        batch_size: Training batch size
        max_epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
    """
    # Download base model
    model_path = download_model(base_model)
    model = load_from_checkpoint(model_path)

    # Prepare data
    train_data = prepare_summarization_data(split="train[:1000]")
    valid_data = prepare_summarization_data(split="validation[:200]")

    # Save data in format expected by COMET
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, "dev.json"), "w") as f:
        json.dump(valid_data, f)

    # Configure training
    model.hparams.learning_rate = learning_rate
    model.hparams.train_data = os.path.join(output_dir, "train.json")
    model.hparams.validation_data = os.path.join(output_dir, "dev.json")
    model.hparams.batch_size = batch_size

    # Set up trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="comet-summarization-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model)

    # Save the final model
    model.save_pretrained(os.path.join(output_dir, "final-model"))

    print(f"Fine-tuned model saved to {output_dir}")
    return os.path.join(output_dir, "final-model")


if __name__ == "__main__":
    model_path = finetune_comet_for_summarization()
    print(f"Use this model path in your calculate_comet_score function: {model_path}")
