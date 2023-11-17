#!/usr/bin/python3

import subprocess
import shutil
import shlex
import re

print("Downloading models via git...")
with open("local-models.yml") as infile:
    for line in infile:
        if not line.startswith("  -") or "/" not in line:
            continue

        model = re.sub(r"^[^a-z0-9]", "", line.strip()).strip()
        if model.startswith("openai/"):
            continue

        model_git_url = f"https://huggingface.co/{model}"
        print(f"Downloading {model_git_url}")
        subprocess.run(shlex.split(f"git clone --depth 1 {model_git_url}"))

        # we don't need this and it saves like 50% of the space
        shutil.rmtree(f"{model}/.git", ignore_errors=True)
