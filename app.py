"""
Rohin Vijay
FastAPI backend for the meme-stock sentiment model
--------------------------------
Loads BERT and tokenizer once, when the server starts and exposes a POST /predict endpoint for making predictions
"""

import model_setup
import os
import json
import shutil
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Pydantic class validates shape of expected data
class PredictRequest(BaseModel):
    # Converts incoming JSON to a text attribute
    text: str

# Loads model, tokenizer, and label map
def loadResources():
    # Defines path to config/tokenizer files
    model_dir = os.path.dirname(model_setup.paths["hf_export/tokenizer.json"])

    # Defines path to HuggingFace model weights for initial download
    downloaded_weights_path = model_setup.paths["hf_export/model.safetensors"]

    # Combines weights and config/tokenizer files to one directory
    weights_path = os.path.join(model_dir, "model.safetensors")

    # Copies downloaded weights path to directory if it doesn't exist
    if not os.path.exists(weights_path):
        shutil.copy(downloaded_weights_path, weights_path)

    # Defines tokenizer and model from directory path
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Switch to evaluation mode to deactivate dropout layers
    model.eval()

    # Loads label map config from JSON file
    with open(model_setup.paths["hf_export/label_map.json"]) as label_map_file:
        label_map = json.load(label_map_file)

    return tokenizer, model, label_map


# Runs the tokenization, inference, and scoring pipeline for a text input
def runPrediction(text: str, tokenizer, model, label_map: dict) -> dict:
    # Converts raw text into the numeric tensors BERT expects
    inputs = tokenizer(text, return_tensors = "pt", truncation = True, max_length = 512, padding = True)

    # Runs model for inference and extracts raw prediction scores for labels
    with torch.no_grad():
        logits = model(**inputs).logits

    # Converts logits to be interpreted as probabilities up to 1.0
    probabilities = F.softmax(logits, dim = -1)[0]

    # Initializes dictionary for label scores
    scores = {}

    # Iterates through each label probability and assigns it to dict
    for i in range(len(probabilities)):
        label = label_map[str(i)]
        score = round(probabilities[i].item(), 4)
        scores[label] = score

    # Pick whichever label has the highest probability
    top_label = max(scores, key = scores.get)

    return {"label": top_label, "scores": scores}


# Initializes FastAPI app to run model and handle http requests
def createApp(tokenizer, model, label_map: dict) -> FastAPI:
    # Instantiates FastAPI application
    app = FastAPI()

    # Allows browsers on other domains to call API
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # Creates a GET endpoint to check server status
    @app.get("/")
    def healthCheck():
        # Confirms the server is running successfully
        return {"status": "ok"}

    # Creates a POST endpoint for making predictions
    @app.post("/predict")
    def predict(request: PredictRequest):
        # Runs prediction requests through model pipeline and returns results
        return runPrediction(request.text, tokenizer, model, label_map)

    return app


# Load the model, tokenizer, and label map once, when this module is imported
tokenizer, model, label_map = loadResources()
# Build the FastAPI app using everything loaded above
app = createApp(tokenizer, model, label_map)