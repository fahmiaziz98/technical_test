import os

MODEL_NAME = "ds4sd/SmolDocling-256M-preview"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "docs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SEPARATOR = "\n\n"
