import os
import torch
from src.data_loader import load_data, prepare_data_splits
from src.model_setup import setup_model
from src.dataset import LoadDataset
from src.training import setup_trainer, save_model
from src.utils import setup_output_directory

def main():
    # Configuration
    encoder_checkpoint = "google/vit-base-patch16-224-in21k"
    decoder_checkpoint = "gpt2"
    max_length = 384
    output_dir = "image-caption-generator"
    
    # Data paths (update these according to your data location)
    projections_path = "path/to/indiana_projections.csv"
    reports_path = "path/to/indiana_PROreports.csv"
    images_path = "path/to/images/images_normalized/"
    
    # Load and prepare data
    print("Loading data...")
    images_captions_df = load_data(projections_path, reports_path)
    train_df, test_df = prepare_data_splits(images_captions_df, images_path)
    
    # Set up model and tokenizers
    print("Setting up model...")
    model, feature_extractor, tokenizer = setup_model(encoder_checkpoint, decoder_checkpoint)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_ds = LoadDataset(train_df, feature_extractor, tokenizer, max_length)
    test_ds = LoadDataset(test_df, feature_extractor, tokenizer, max_length)
    
    # Set up training
    print("Setting up trainer...")
    setup_output_directory(output_dir)
    trainer = setup_trainer(model, train_ds, test_ds, feature_extractor, output_dir)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    save_model(model, os.path.join(output_dir, "model_final.pt"))
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
