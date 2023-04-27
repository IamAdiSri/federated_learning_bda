import os
import threading

import json
import io

import base64
import hashlib

from flask import Flask, request, Response

from .fl import evaluate_model, map_func, reduce_func

from google.cloud import storage
from pyspark import SparkConf
from pyspark.sql import SparkSession

app = Flask(__name__)

BUCKET_NAME = os.getenv("BUCKET_NAME") if os.getenv("BUCKET_NAME") else "bdastorage"
BUCKET_PREFIX = "models" # folder in bucket where models are stored
MODEL_VER = 0
MODEL_DIR = os.getenv("MODEL_DIR") if os.getenv("MODEL_DIR") else "/home/adisri/bda/proj/ops/models" # folder on vm where models from the bucket are downloaded to
DATA_DIR = os.getenv("DATA_DIR") if os.getenv("DATA_DIR") else "/home/adisri/bda/proj/ops/splits" # folder on vm where data from the bucket is downloaded to
AVERAGING_ALGORITHM = "mean"
FETCH_COUNTER = 0

# Set up pyspark
sparkConf = SparkConf().setAppName("GCSFilesRead").set("spark.executor.memory", "5g").set("spark.driver.memory", "5g")
# sparkConf = SparkConf().setAppName("GCSFilesRead").set("spark.executor.memory", "5g").set("spark.driver.memory", "5g").set("spark.pyspark.python", "python3.9")
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
spark._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile", "credentials.json")

# set up google cloud storage
storage_client = storage.Client.from_service_account_json("credentials.json")
bucket = storage_client.get_bucket(BUCKET_NAME)

@app.route("/")
def index():
    return """
    <h1> Flagging Hate Speech and Cyber-Bullying in Private Chats</h1>
    <p> Using distributed ML and federated learning techniques to preserve privacy when learning from private data</p>
    """

@app.route("/tracker")
def tracker():
    """
    Track model progress as federated learning commences.
    """
    pass

@app.route("/fetch_model/<int:version>", methods=["GET"])
def fetch_model(version):
    """
    Fetch the most recent global model as a zip file.
    """
    global FETCH_COUNTER

    if version > MODEL_VER:
        print(f"SERVER LOG: Version requested is unavailable")
        return Response(
            response = json.dumps({
                "resource": f"global_v{version}",
                "details": "Requested version is unavailable"
            }),
            status = 404
        )
    
    try:
        GLOBAL_MODEL = f"{MODEL_DIR}/global_v{MODEL_VER}"
        # TODO: fetch global model from bucket
        with open(f"{GLOBAL_MODEL}.zip", 'rb') as f:
            data = f.read()

    except FileNotFoundError:
        if os.path.isdir(f"{GLOBAL_MODEL}"):
            os.system(f"zip {GLOBAL_MODEL}.zip {GLOBAL_MODEL}/*")
            print(f"SERVER LOG: Found and zipped files for model {GLOBAL_MODEL}")
            # TODO: fetch global model from bucket
            with open(f"{GLOBAL_MODEL}.zip", 'rb') as f:
                data = f.read()
        else:
            print(f"SERVER LOG: Could not find model {GLOBAL_MODEL}")
            return Response(
                response = json.dumps({
                    "resource": GLOBAL_MODEL,
                    "details": "Could not find resource"
                }),
                status = 500
        )
    
    FETCH_COUNTER += 1
    print(f"SERVER LOG: {FETCH_COUNTER} finetune requests pending")

    print(f"SERVER LOG: Sending model {GLOBAL_MODEL} to client")
    return Response(
        response = io.BytesIO(data),
        status = 200,
        headers = {"Content-Type": "application/zip"}
    )

@app.route("/upload_model", methods=["POST"])
def upload_model():
    """
    Upload a finetuned model.
    """
    global FETCH_COUNTER

    r = request.get_json()

    # decode data to binary object
    model = base64.b64decode(r["model"])

    # TODO: upload model to bucket
    LOCAL_MODEL_DIR = f"{MODEL_DIR}/local_v{MODEL_VER}"

    # get hash value of object
    hash = hashlib.sha256(model).hexdigest()

    if not os.path.isdir(LOCAL_MODEL_DIR):
        os.makedirs(f"{LOCAL_MODEL_DIR}")
        print(f"SERVER LOG: Created directory {LOCAL_MODEL_DIR}")

    with open(f"{LOCAL_MODEL_DIR}/{hash}.zip", "wb") as f:
        f.write(io.BytesIO(model))

    FETCH_COUNTER -= 1
    print(f"SERVER LOG: {FETCH_COUNTER} finetune requests pending")

    print(f"SERVER LOG: Saved client model as {LOCAL_MODEL_DIR}/{hash}.zip")
    # trigger averaging if fetch counter reaches 0
    if FETCH_COUNTER == 0:
        print("SERVER LOG: Received all client models")
        if AVERAGING_ALGORITHM == "mean":
            # setting daemon=True means the thread will exit if the server shuts down
            thread = threading.Thread(target=average_models, args=[], daemon=True)
            thread.start()
            print("SERVER LOG: Spawned a thread to compute average of local models")
        if AVERAGING_ALGORITHM == "mean":
            # setting daemon=True means the thread will exit if the server shuts down
            thread = threading.Thread(target=weight_average_models, args=[], daemon=True)
            thread.start()
            print("SERVER LOG: Spawned a thread to compute weighted average of local models")

    return Response(
        response = json.dumps({
            "resource": f"{LOCAL_MODEL_DIR}/{hash}.zip",
            "details": f"Successfully saved client model"
        }),
        status = 200,
        headers = {"Content-Type": "application/json"}
    )

def average_models():
    """
    Run federated averaging to combine the finetuned models.
    """
    # TODO: delegate task to spark cluster
    model_loc = f"{MODEL_DIR}/global_v{MODEL_VER}"
    data_loc = f"{DATA_DIR}/test.csv"

    MODEL_VER += 1
    evaluate_model(model_loc, data_loc)
    return

def weight_average_models():
    """
    Run weighted federated averaging to combine the finetuned models.
    """
    # TODO: delegate task to spark cluster

    MODEL_VER += 1
    evaluate_model()
    return

def bucket_list(dir_loc):
    blobs = bucket.list_blobs(prefix=dir_loc)
    return [b.split('/')[-1] if not b.name.endswith('/') 
            else b.split('/')[-2]+'/' for b in blobs]

def bucket_write(file_loc, tgt_loc):
    blob = bucket.blob(file_loc)
    blob.upload_from_filename(tgt_loc)

def bucket_read(tgt_loc, file_loc):
    blob = bucket.blob(tgt_loc)
    blob.download_to_filename(file_loc)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)