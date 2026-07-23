#!/usr/bin/env bash
set -e
apt-get install -y poppler-utils
pip install -r requirements.txt
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
