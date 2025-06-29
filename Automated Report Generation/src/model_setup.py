from transformers import (
    AutoFeatureExtractor, 
    AutoTokenizer, 
    VisionEncoderDecoderModel
)

def setup_model(encoder_checkpoint, decoder_checkpoint):
    """
    Set up the vision-language model and tokenizers.
    
    Args:
        encoder_checkpoint (str): Name or path of the vision encoder model
        decoder_checkpoint (str): Name or path of the text decoder model
        
    Returns:
        tuple: (model, feature_extractor, tokenizer)
    """
    # Initialize feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_checkpoint, 
        decoder_checkpoint
    )
    
    # Configure model parameters
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.num_beams = 4
    
    return model, feature_extractor, tokenizer
