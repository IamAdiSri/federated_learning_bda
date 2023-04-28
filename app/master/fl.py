import os

from google.cloud import storage

import pandas as pd
import numpy as np

import torch
from transformers import DistilBertForSequenceClassification
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report

def evaluate_model(model_loc, data_loc):
    """
    Evaluate merged model.
    """
    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1)

    # Create a ClassificationModel
    model = ClassificationModel(
        "distilbert", model_loc, args=model_args, use_cuda=False
    )

    # Preparing eval data
    eval_df = pd.read_csv(data_loc)
    eval_df.columns = ["text", "labels"]

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    # print results
    preds = np.argmax(model_outputs, axis=1)
    print(classification_report(eval_df.labels, preds))
    
    return
    
def map_func(addr, w, s):
    """
    weight and divide each model
    """
    # client objects cannot be defined outside the scope of the function
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("bdastorage")
    blob = bucket.get_blob(addr)
    name = blob.name.split('/')[-1]
    blob.download_to_filename(name)

    os.system(f"unzip {name}")
    model = ClassificationModel("distilbert", name, use_cuda=False)
    with torch.no_grad():
        layers = model.model.state_dict().keys()
        for layer in layers:
            model.model.state_dict()[layer].data.copy_(model.model.state_dict()[layer].data*w/s)
    return model.model
    
def reduce_func(model1, model2):
    """
    add all weighted models
    """
    with torch.no_grad():
        added = DistilBertForSequenceClassification(config=model1.config)
        layers = model1.state_dict().keys()
        for layer in layers:
            added.state_dict()[layer].data.copy_(model1.state_dict()[layer].data + model2.state_dict()[layer].data)
        return added