import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(projections_path, reports_path):
    """
    Load and preprocess the chest X-ray data from CSV files.
    
    Args:
        projections_path (str): Path to the projections CSV file
        reports_path (str): Path to the reports CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe with image paths and captions
    """
    df2 = pd.read_csv(projections_path)
    df1 = pd.read_csv(reports_path)
    
    # Filter for frontal projections only
    df2 = df2[df2['projection'] == 'Frontal']
    
    # Create image-caption pairs
    images_captions_df = pd.DataFrame({'imgs': [], 'captions': []})
    for i in range(len(df2)):
        uid = df2.iloc[i]['uid']
        image = df2.iloc[i]['filename']
        index = df1.loc[df1['uid'] == uid]
        
        if not index.empty:    
            index = index.index[0]
            caption = df1.iloc[index]['findings']
            if type(caption) == float:
                continue 
            images_captions_df = pd.concat(
                [images_captions_df, pd.DataFrame([{'imgs': image, 'captions': caption}])], 
                ignore_index=True
            )
    
    return images_captions_df

def prepare_data_splits(df, images_path, test_size=0.2, random_state=42):
    """
    Prepare train and test splits from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe with image paths and captions
        images_path (str): Base path for the images
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (train_df, test_df) containing train and test splits
    """
    # Add full image paths
    df['imgs'] = images_path + df['imgs']
    
    # Create train-test split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        shuffle=True, 
        random_state=random_state
    )
    
    return train_df, test_df
