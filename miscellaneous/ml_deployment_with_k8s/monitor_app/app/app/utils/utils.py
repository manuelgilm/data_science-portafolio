from pathlib import Path 

def get_root_dir()->Path:
    """
    Gets root dir
    """

    return Path(__file__).parent.parent.parent