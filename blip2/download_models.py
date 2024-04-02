# Setup models to queue download during build
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Add model types to download
model_types = ["Salesforce/blip2-opt-2.7b"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download models
for model_type in model_types:
    processor = AutoProcessor.from_pretrained(model_type)
    model = Blip2ForConditionalGeneration.from_pretrained(model_type, torch_dtype=torch.float16)
    model.to(device)
