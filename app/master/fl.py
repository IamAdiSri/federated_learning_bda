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

def map_func(model):
    # divide operation
    with torch.no_grad():
        layers = model.state_dict().keys()
        for layer in layers:
            model.state_dict()[layer].data.copy_(model.state_dict()[layer].data / 2)
        return model
    
def reduce_func(model1, model2, weights=(1, 1)):
    # add operation
    with torch.no_grad():
        added = DistilBertForSequenceClassification(config=model1.config)
        layers = model1.state_dict().keys()
        for layer in layers:
            added.state_dict()[layer].data.copy_(weights[0]*model1.state_dict()[layer].data + weights[1]*model2.state_dict()[layer].data)
        return added