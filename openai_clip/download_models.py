import clip
device = "cuda" if torch.cuda.is_available() else "cpu"

for model_name in clip.available_models():
    model, preprocess = clip.load(model_name, device=device)
