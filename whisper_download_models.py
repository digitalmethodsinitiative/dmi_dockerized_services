from whisper import _download, _MODELS

models = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large"]

for model in models:
    print(f"Downloading {model} model...\n")
    _download(_MODELS[model], "/root/.cache/whisper", False)
