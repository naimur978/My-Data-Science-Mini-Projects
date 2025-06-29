from torch.utils.data import Dataset
from PIL import Image

class LoadDataset(Dataset):
    """
    Dataset class for loading image-caption pairs.
    """
    def __init__(self, df, feature_extractor, tokenizer, max_length=384):
        """
        Initialize the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing image paths and captions
            feature_extractor: Vision feature extractor
            tokenizer: Text tokenizer
            max_length (int): Maximum length for captions
        """
        self.images = df['imgs'].values
        self.captions = df['captions'].values
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            dict: Dictionary containing processed inputs
        """
        inputs = dict()

        # Load and process the image
        image_path = str(self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.feature_extractor(images=image, return_tensors='pt')

        # Process the caption
        caption = self.captions[idx]
        labels = self.tokenizer(
            caption, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt',
        )['input_ids'][0]
        
        # Store processed inputs
        inputs['pixel_values'] = image['pixel_values'].squeeze()   
        inputs['labels'] = labels
        
        return inputs
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of items in the dataset
        """
        return len(self.images)
