import os
import time
import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

from pipeline.constant import MODEL_NAME
from pipeline.utils import (
    clean_markdown_text,
    get_compute_capability,
    get_device,
    setup_logging
)
logger = setup_logging()

def load_model_and_processor(device: str) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """
    Load the vision model and processor for OCR processing.

    Args:
        device (str): The device to load the model on ('cuda' or 'cpu')

    Returns:
        tuple: A tuple containing:
            - processor (AutoProcessor): The loaded processor
            - model (AutoModelForVision2Seq): The loaded model

    Note:
        The model will use bfloat16 precision on CUDA devices and float32 on CPU.
        Flash attention will be used on CUDA devices with compute capability >= 8.
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        _attn_implementation="flash_attention_2" if device == "cuda" and get_compute_capability() >= 8 else "eager",
    ).to(device)
    return processor, model

def process_image(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    device: str
) -> tuple[str, str, float]:
    """
    Process a single image through the OCR pipeline.

    Args:
        image (Image.Image): The input image to process
        processor (AutoProcessor): The loaded processor
        model (AutoModelForVision2Seq): The loaded model
        device (str): The device to run inference on

    Returns:
        tuple: A tuple containing:
            - text_clean (str): The cleaned extracted text
            - markdown (str): The markdown formatted text
            - generation_time (float): Time taken for generation in seconds
    """
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"}, {"type": "text", "text": "Convert this page to docling."}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generation_time = time.time() - start_time

    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids, 
        skip_special_tokens=False
    )[0].replace("<end_of_utterance>", "").strip()

    doc_tags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doc_tags_doc)
    markdown = doc.export_to_markdown()

    text_clean = clean_markdown_text(doctags)

    return text_clean, markdown, generation_time

def process_pdf(
    pdf_path: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    device: str
) -> tuple[str, str]:
    """
    Process a single PDF file and extract text from all its pages.

    Args:
        pdf_path (str): Path to the PDF file
        processor (AutoProcessor): The loaded processor
        model (AutoModelForVision2Seq): The loaded model
        device (str): The device to run inference on

    Returns:
        tuple: A tuple containing:
            - texts (list): List of extracted text from each page
            - markdowns (list): List of markdown formatted text from each page
    """
    filename = os.path.basename(pdf_path)
    images = convert_from_path(pdf_path)
    texts, markdowns = [], []

    logger.info(f"Processing: {filename}")

    for i, image in enumerate(images):
        try:
            text, markdown, gen_time = process_image(image, processor, model, device)
            texts.append(text)
            markdowns.append(markdown)
            logger.info(f"Page {i+1} of {filename} processed in {gen_time:.2f} s")
            
        except Exception as e:
            logger.error(f"Error on page {i+1} of {filename}: {e}")

    return texts, markdowns

def process_all_pdfs(input_dir: str) -> tuple[list, list]:
    """
    Process all PDF files in the specified directory.

    This function:
    1. Sets up the device (CPU/CUDA)
    2. Loads the model and processor
    3. Processes each PDF file in the input directory
    4. Combines the results from all PDFs

    Args:
        input_dir (str): Directory containing PDF files to process

    Returns:
        tuple: A tuple containing:
            - all_texts (list): List of extracted text from all PDFs
            - all_markdowns (list): List of markdown formatted text from all PDFs
    """
    device = get_device()
    processor, model = load_model_and_processor(device)

    all_texts = []
    all_markdowns = []

    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(input_dir, file)
            texts, markdowns = process_pdf(full_path, processor, model, device)
            # Combine all pages in one document
            all_texts.append("\n".join(texts))
            all_markdowns.append("\n".join(markdowns))

    return all_texts, all_markdowns

# if __name__ == "__main__":
#     texts, markdowns = process_all_pdfs(INPUT_DIR)
#     save_combined_output(texts, markdowns)