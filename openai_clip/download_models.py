import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

for model_name in clip.available_models():
    print(f"Downloading {model_name} model...\n")
    model, preprocess = clip.load(model_name, device=device)
