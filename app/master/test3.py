# checks if simpletransformers can be loaded on workers
# and models can be averaged
import os
import time
from google.cloud import storage
from pyspark import SparkConf
from pyspark.sql import SparkSession
from simpletransformers.classification import ClassificationModel
from transformers import DistilBertForSequenceClassification
import torch

storage_client = storage.Client()
bucket = storage_client.get_bucket("bdastorage")

sparkConf = SparkConf()
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

def mapp(addr, s):
    # client objects cannot be defined outside the scope of the function
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("bdastorage")
    blob = bucket.get_blob(addr)
    blob.download_to_filename("model.zip")
    os.system("unzip model.zip")
    model = ClassificationModel("distilbert", "global_v0", use_cuda=False)
    with torch.no_grad():
        layers = model.model.state_dict().keys()
        for layer in layers:
            model.model.state_dict()[layer].data.copy_(model.model.state_dict()[layer].data/s)
    return model.model

def reduce(model1, model2):
    with torch.no_grad():
        added = DistilBertForSequenceClassification(config=model1.config)
        layers = model1.state_dict().keys()
        for layer in layers:
            added.state_dict()[layer].data.copy_(model1.state_dict()[layer].data + model2.state_dict()[layer].data)
    return added

ct = time.time()
s = 2
addr = "models/global_v0.zip"
rdd = spark.sparkContext.parallelize([addr for _ in range(s)])
avg_model = rdd.map(lambda x: mapp(addr, s)).reduce(reduce)

os.makedirs("averaged")
avg_model.save_pretrained("averaged") # saving averaged model
os.system("zip -r averaged.zip averaged/")

blob = bucket.blob("models/averaged")
blob.upload_from_filename("averaged.zip")

print(avg_model, time.time()-ct)

