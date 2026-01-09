import yaml
from pathlib import Path

def load_config(path : str):
    path = Path(path)
    
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config