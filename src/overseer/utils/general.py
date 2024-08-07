import os
from overseer.utils.logging import OverseerLogger

logger = OverseerLogger().get_logger("GeneralUtils")

def fix_path(path: str, add_tif: bool = True) -> str:
    """
    Fix and validate the given path.

    Args:
        path (str): The path to fix and validate.
        add_tif (bool): Whether to add .tif extension if missing. Default is True.

    Returns:
        str: The fixed and validated path.
    """
    # Remove any surrounding quotes or parentheses
    path = path.strip("'\"()")
    
    # Add .tif extension if it's missing and add_tif is True
    if add_tif and not path.lower().endswith('.tif'):
        path += '.tif'
    
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    # Check if the path exists
    if not os.path.exists(abs_path):
        logger.warning(f"Path does not exist: {abs_path}")
        return ""
    
    return abs_path