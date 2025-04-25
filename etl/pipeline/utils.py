import os
import re
import torch
import logging
from pipeline.constant import OUTPUT_DIR, SEPARATOR

def setup_logging() -> logging.Logger:
    """Configure and set up basic logging settings.

    Sets up logging with INFO level and a specific format that includes timestamp,
    log level, and message.

    Returns:
        logging.Logger: Configured logger instance for the module.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_device() -> str:
    """Determine the available computing device.

    Returns:
        str: Either "cuda" if a CUDA-capable GPU is available, or "cpu" otherwise.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_compute_capability(device: torch.device = None) -> int:
    """Get the compute capability of the CUDA device.

    Args:
        device (torch.device, optional): The torch device to check. If None,
            returns the minimum compute capability across all available devices.

    Returns:
        int: The compute capability major version number. Returns 0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        if device is None:
            return min(torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count()))
        return torch.cuda.get_device_properties(device).major
    return 0

def clean_markdown_text(text: str) -> str:
    """Clean and format markdown text by removing unwanted elements and normalizing spacing.

    This function performs several cleaning operations:
    - Removes page footer tags and their content
    - Strips HTML tags
    - Normalizes line breaks and spacing
    - Removes duplicate punctuation
    - Cleans up page numbers
    - Preserves paragraph structure

    Args:
        text (str): The input markdown text to be cleaned.

    Returns:
        str: The cleaned and formatted text.
    """
    # clean text, mengekstrak secara manual karena
    # ketika mengeksport pake md beberapa text hilang
    data = re.sub(r'<page_footer>.*?</page_footer>', '', text, flags=re.DOTALL)
    text_clean = re.sub(r'</?[^>]+>', '', data)
    
    lines = text_clean.splitlines()
    cleaned_lines = []
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            cleaned_lines.append("")
            continue
        cleaned_lines.append(clean_line)

    cleaned_text = "\n".join(cleaned_lines)

    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)      
    cleaned_text = re.sub(r'\n +', '\n', cleaned_text)      
    cleaned_text = re.sub(r' +\n', '\n', cleaned_text)      

    cleaned_text = re.sub(r':{2,}', ':', cleaned_text)
    cleaned_text = re.sub(r'·{2,}', '·', cleaned_text)

    cleaned_text = re.sub(r'\n\d{1,3}\n', '\n', cleaned_text)

    return cleaned_text.strip()

def save_combined_output(texts: str, markdowns: str) -> None:
    """Save combined text and markdown outputs to files.

    Creates an output directory if it doesn't exist and saves two files:
    - combined_output.txt: Contains joined text content
    - combined_output.md: Contains joined markdown content

    Args:
        texts (list): List of text contents to be combined
        markdowns (list): List of markdown contents to be combined

    Returns:
        None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "combined_output.txt"), "w", encoding="utf-8") as f_txt:
        f_txt.write(SEPARATOR.join(texts))

    with open(os.path.join(OUTPUT_DIR, "combined_output.md"), "w", encoding="utf-8") as f_md:
        f_md.write(SEPARATOR.join(markdowns))

    logging.info("Combined outputs saved successfully.")
    print("Combined outputs saved successfully.")