import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluate import load
from bert_score import BERTScorer

def generate_caption(model, image_tensor, tokenizer, max_length=384):
    """
    Generate a caption for a single image.
    
    Args:
        model: The trained model
        image_tensor: Processed image tensor
        tokenizer: Text tokenizer
        max_length (int): Maximum length of generated caption
        
    Returns:
        str: Generated caption
    """
    model.eval()
    with torch.no_grad():
        out = model.generate(
            image_tensor.unsqueeze(0).to('cuda'),
            num_beams=4,
            max_length=max_length
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def evaluate_model(model, test_ds, tokenizer, num_samples=250):
    """
    Evaluate the model using BLEU score and BERTScore.
    
    Args:
        model: The trained model
        test_ds: Test dataset
        tokenizer: Text tokenizer
        num_samples (int): Number of samples to evaluate
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    DS = []
    GPT = []
    model.eval()
    
    for i in tqdm(range(num_samples)):
        inputs = test_ds[i]['pixel_values']
        with torch.no_grad():
            out = model.generate(
                inputs.unsqueeze(0).to('cuda'),
                num_beams=4,
                max_length=384
            )
        
        y_hat = tokenizer.decode(test_ds[i]['labels'], skip_special_tokens=True)
        DS.append(y_hat)
        
        y_pred = tokenizer.decode(out[0], skip_special_tokens=True)
        GPT.append(y_pred)
    
    # Calculate BLEU score
    bleu = load("bleu")
    bleu_results = bleu.compute(predictions=GPT, references=DS)
    
    # Calculate BERTScore
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(GPT, DS)
    bert_results = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
    
    return {
        'bleu': bleu_results,
        'bert_score': bert_results,
        'reference_captions': DS,
        'generated_captions': GPT
    }

def visualize_prediction(model, image_tensor, tokenizer, max_length=384):
    """
    Visualize an image and its generated caption.
    
    Args:
        model: The trained model
        image_tensor: Processed image tensor
        tokenizer: Text tokenizer
        max_length (int): Maximum length of generated caption
    """
    caption = generate_caption(model, image_tensor, tokenizer, max_length)
    print("Generated Caption:", caption)
    
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(torch.permute(image_tensor, (1, 2, 0)))
    plt.show()
