#!/usr/bin/env bash
set -e
which pdftoppm || echo "poppler not found"
pip install -r requirements.txt
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
