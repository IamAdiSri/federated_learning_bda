import subprocess
import os

subprocess.call(['python','-m','pip', 'install', 'simpletransformers'])
subprocess.call(['python','-m','pip', 'install', 'transformers'])
subprocess.call(['python','-m','pip', 'install', 'torch'])
subprocess.call(['python','-m','pip', 'install', 'jupyter'])
subprocess.call(['python','-m','pip', 'install', 'ipywidgets'])

from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark import SparkConf
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import DistilBertForSequenceClassification
import torch

# Create a Spark session
# Create a service acc and download credentials as json. This is needed for bucket access. Save them inside cloud shell and pass Storage Legacy Bucket Reader,Storage Object Admin, Storage Object Viewer
sparkConf = SparkConf().setAppName("GCSFilesRead").set("spark.executor.memory", "5g").set("spark.driver.memory", "5g")
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

# Set the Google Cloud credentials
spark._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile","credentials.json")

# Get the Google Cloud Storage client
storage_client = storage.Client.from_service_account_json('credentials.json')

from pathlib import Path
bucket_name = 'bdastorage'
prefix = 'models/stmodel/'
dl_dir = './models/stmodel/'

bucket = storage_client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    file_split = blob.name.split("/")
    print("file_split",file_split)
    directory = "./" + "/".join(file_split[0:-1])
    print("directory", directory, "blob.name", blob.name)
    Path(directory).mkdir(parents=True, exist_ok=True)
    print("SAVING INSIDE", directory+"/"+file_split[-1])
    blob.download_to_filename(directory+"/"+file_split[-1]) 

model_args = ClassificationArgs(num_train_epochs=1)
model = ClassificationModel(
    "distilbert",
    dl_dir,
    num_labels=2,
    use_cuda=False,
    args=model_args
)
print("MODEL")
print(model.model)
# Define the map and reduce functions
def map_func(model):
    # import subprocess, os
    # print(">>>", os.popen("which python").read())
    # subprocess.call(['python','-m','pip', 'install', 'simpletransformers'])
    # subprocess.call(['python','-m','pip', 'install', 'transformers'])
    # subprocess.call(['python','-m','pip', 'install', 'torch'])
    # subprocess.call(['python','-m','pip', 'install', 'jupyter'])
    # subprocess.call(['python','-m','pip', 'install', 'ipywidgets'])
    from transformers import DistilBertForSequenceClassification
    import torch
    # divide operation
    with torch.no_grad():
        layers = model.state_dict().keys()
        for layer in layers:
            model.state_dict()[layer].data.copy_(model.state_dict()[layer].data / 2)
        return model

def reduce_func(model1, model2):
    # import subprocess, os
    # print(">>>", os.popen("which python").read())
    # subprocess.call(['python','-m','pip', 'install', 'simpletransformers'])
    # subprocess.call(['python','-m', 'pip', 'install', 'transformers'])
    # subprocess.call(['python','-m', 'pip', 'install', 'torch'])
    # subprocess.call(['python','-m','pip', 'install', 'jupyter'])
    # subprocess.call(['python','-m','pip', 'install', 'ipywidgets'])
    from transformers import DistilBertForSequenceClassification
    import torch
    # add operation
    with torch.no_grad():
        added = DistilBertForSequenceClassification(config=model1.config)
        layers = model1.state_dict().keys()
        for layer in layers:
            added.state_dict()[layer].data.copy_(model1.state_dict()[layer].data + model2.state_dict()[layer].data)
        return added

# Create the PySpark RDD
rdd = spark.sparkContext.parallelize([model, model], numSlices=10000)

# Apply the map and reduce functions to the RDD
mapped_rdd = rdd.map(map_func)
averaged_model = mapped_rdd.reduce(reduce_func)

# Save the averaged model to a file
averaged_model.save_pretrained('stmodelAvg')
print(averaged_model.model)

# Upload the averaged model to GCS
blob = bucket.blob('models/stmodelAvg')
blob.upload_from_filename('stmodelAvg')
