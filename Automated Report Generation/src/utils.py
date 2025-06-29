import os

def create_directory(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): Path to the directory to create
    """
    os.makedirs(directory, exist_ok=True)

def remove_directory_contents(directory):
    """
    Remove all contents of a directory.
    
    Args:
        directory (str): Path to the directory to clean
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                remove_directory_contents(item_path)
                os.rmdir(item_path)
        except Exception as e:
            print(f"Error removing {item_path}: {e}")
            
def setup_output_directory(output_dir):
    """
    Set up the output directory for model checkpoints.
    
    Args:
        output_dir (str): Path to the output directory
    """
    if os.path.exists(output_dir):
        remove_directory_contents(output_dir)
        os.rmdir(output_dir)
    create_directory(output_dir)
