from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)

def setup_trainer(model, train_ds, test_ds, feature_extractor, output_dir="image-caption-generator"):
    """
    Set up the trainer for model training.
    
    Args:
        model: The model to train
        train_ds: Training dataset
        test_ds: Test dataset
        feature_extractor: Vision feature extractor
        output_dir (str): Directory to save model checkpoints
        
    Returns:
        Seq2SeqTrainer: Configured trainer instance
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=4,
        save_strategy='epoch',
        report_to='none',
        gradient_accumulation_steps=4
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        data_collator=default_data_collator,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
    )
    
    return trainer

def save_model(model, save_path):
    """
    Save the model's state dictionary.
    
    Args:
        model: The model to save
        save_path (str): Path where to save the model
    """
    import torch
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path):
    """
    Load a saved model's state dictionary.
    
    Args:
        model: The model to load weights into
        load_path (str): Path to the saved model weights
        
    Returns:
        The model with loaded weights
    """
    import torch
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    return model
