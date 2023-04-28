import os
import threading
import time

import json
import io

import base64
import hashlib

from flask import Flask, request, Response

from fl import evaluate_model, map_func, reduce_func

from google.cloud import storage
from pyspark import SparkConf
from pyspark.sql import SparkSession

app = Flask(__name__)

BUCKET_NAME = os.getenv("BUCKET_NAME") if os.getenv("BUCKET_NAME") else "bdastorage"
AVG_ALGO = os.getenv("AVG_ALGO") if os.getenv("AVG_ALGO") else "mean"
MODEL_VER = None
STATUS = None
FETCH_COUNTER = 0

# Set up pyspark
sparkConf = SparkConf()
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

# set up google cloud storage
storage_client = storage.Client()
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
    Fetch the most global model of the specified version as a zip file.
    """
    global FETCH_COUNTER

    if version != MODEL_VER:
        print(f"SERVER LOG: Version requested is unavailable")
        return Response(
            response = json.dumps({
                "resource": f"global_v{version}",
                "details": "Requested version is unavailable"
            }),
            status = 404
        )
    
    if os.path.exists(f"global_v{version}.zip"):
        print(f"SERVER LOG: Found global_v{version}.zip")
    elif os.path.exists(f"global_v{version}/"):
        print(f"SERVER LOG: Found directory global_v{version}")
        os.system(f"zip -r global_v{version}.zip global_v{version}")
    else:
        blob = bucket.blob(f"models/global_v{version}.zip")
        if blob.exists():
            print(f"SERVER LOG: Found global_v{version} on bucket")
            blob.download_to_filename(f"global_v{version}.zip")
        else:
            print(f"SERVER LOG: Could not find global_v{version}")
            return Response(
                response = json.dumps({
                    "resource": f"global_v{version}",
                    "details": "Could not find resource"
                }),
                status = 500
            )

    with open(f"global_v{version}.zip", 'rb') as f:
        data = f.read()
    
    if STATUS["merging"] and STATUS["evaluating"]:
        # If client models are being merged or evaluated, do not 
        # allow a new merge to start by updating the FETCH_COUNTER
        print(f"SERVER LOG: Global model is being merged/evaluated")
    else:
        FETCH_COUNTER += 1
    print(f"SERVER LOG: {FETCH_COUNTER} finetune requests pending")

    print(f"SERVER LOG: Sending model global_v{version}.zip to client")
    return Response(
        response = io.BytesIO(data),
        status = 200,
        headers = {"Content-Type": "application/zip"}
    )

@app.route("/upload_model/<int:version>", methods=["POST"])
def upload_model(version):
    """
    Upload a finetuned model.
    """
    global FETCH_COUNTER
    global STATUS

    if version != MODEL_VER:
        print(f"SERVER LOG: Upload request is for a bad version")
        return Response(
            response = json.dumps({
                "resource": f"local_v{version}",
                "details": "Upload request is for a bad version"
            }),
            status = 404
        )
    elif STATUS["merging"] or STATUS["evaluating"]:
        print(f"SERVER LOG: Upload request denied as the global model is being merged/evaluated")
        return Response(
            response = json.dumps({
                "resource": f"local_v{version}",
                "details": "Upload request denied"
            }),
            status = 404
        )

    r = request.get_json()

    # decode data to binary object
    model = base64.b64decode(r["model"])
    sample_size = int(r["sample_size"])

    # get hash value of object
    hash = hashlib.sha256(model).hexdigest()

    blob = bucket.blob(f"models/local_v{version}_{hash}.zip")
    blob.upload_from_file(io.BytesIO(model))
    print(f"SERVER LOG: Saved local_v{version}_{hash}.zip to bucket")

    # update weights in status variable
    STATUS["sample_sizes"][hash] = sample_size
    fl_status(op="write")

    FETCH_COUNTER -= 1
    print(f"SERVER LOG: {FETCH_COUNTER} finetune requests pending")

    # trigger averaging if fetch counter reaches 0
    if FETCH_COUNTER == 0:
        print("SERVER LOG: Received all client models, commencing model averaging...")

        # setting daemon=True means the thread will exit if the server shuts down
        thread = threading.Thread(target=average_models, args=[AVG_ALGO], daemon=True)
        thread.start()
        print(f"SERVER LOG: Spawned a thread to compute {AVG_ALGO} of local models")

    return Response(
        response = json.dumps({
            "resource": f"local_v{version}_{hash}.zip",
            "details": f"Successfully saved client model"
        }),
        status = 200,
        headers = {"Content-Type": "application/json"}
    )

def average_models(algo):
    """
    Run federated averaging to combine the finetuned models.
    """
    global MODEL_VER
    global STATUS

    # merge on spark cluster
    STATUS["merging"] = True
    blobs = bucket.list_blobs(prefix="models")
    models_list = [b.name for b in blobs if b.name.endswith(".zip") 
                  and b.name.split('/')[-1].startswith(f"local_v{MODEL_VER}_")]
    if algo=="mean":
        weights = {h:1 for h in STATUS["sample_sizes"]}
    elif algo=="wmean":
        s = sum(STATUS["sample_sizes"].values())
        weights = {h:STATUS["sample_sizes"]/s for h in STATUS["sample_sizes"]}
    total = len(models_list)

    print("SERVER LOG: Starting MapReduce...")
    curr_time = time.time()
    rdd = spark.sparkContext.parallelize(models_list)
    avg_model = rdd.map(lambda m: map_func(m, weights[m.split('_')[2].split('.')[0]], total)).reduce(reduce_func)
    print(f"SERVER LOG: MapReduce completed. Time taken: {time.time()-curr_time}")

    os.mkdir(f"global_v{MODEL_VER+1}")
    avg_model.save_pretrained(f"global_v{MODEL_VER+1}")
    print("SERVER LOG: Saved model binaries...")

    if not os.path.exists(f"configs"):
        if not os.path.exists(f"configs.zip"):
            blob = bucket.blob("models/configs.zip")
            if not blob.exists():
                print(f"SERVER LOG: Could not find model config constants on the bucket")
            else:
                blob.download_to_filename("configs.zip")
        os.system(f"unzip configs.zip")

    os.system(f"cp configs/* global_v{MODEL_VER+1}")
    os.system(f"zip -r global_v{MODEL_VER+1}.zip global_v{MODEL_VER+1}")
    print("SERVER LOG: Added config files to model package...")

    blob = bucket.blob(f"models/global_v{MODEL_VER+1}.zip")
    blob.upload_from_filename(f"global_v{MODEL_VER+1}.zip")
    print("SERVER LOG: Saved averaged model to bucket.")
    STATUS["merging"] = False

    print(f"SERVER LOG: Starting evaluation on model global_v{MODEL_VER+1}...")
    STATUS["evaluating"] = True
    if not os.path.exists("test.csv"):
        blob = bucket.get_blob("data/test.csv")
        blob.download_to_filename("test.csv")
    # evaluate_model(f"global_v{MODEL_VER+1}", "test.csv")
    STATUS["evaluating"] = False
    print(f"SERVER LOG: Evaluation completed. Check the tracking page to see scores.")
    
    MODEL_VER += 1

    STATUS["version"] = MODEL_VER
    STATUS["sample_sizes"] = {}
    fl_status(op="write")
    print(f"SERVER LOG: Global model version updated. Global model status updated.")

    return

def fl_status(op="read"):
    global STATUS
    blob = bucket.get_blob("models/fl.json")
    if op == "read":
        STATUS = json.loads(blob.download_as_text())
    elif op=="write":
        blob.upload_from_string(json.dumps(STATUS))
    return

if __name__ == '__main__':
    fl_status(op="read")
    MODEL_VER = STATUS["version"]
    print(f"SERVER LOG: BUCKET_NAME={BUCKET_NAME}")
    print(f"SERVER LOG: AVG_ALGO={AVG_ALGO}")
    print(f"SERVER LOG: STATUS={STATUS}")
    print(f"SERVER LOG: MODEL_VER={MODEL_VER}")
    print(f"SERVER LOG: FETCH_COUNTER={FETCH_COUNTER}")
    app.run(host="0.0.0.0", port=5000, debug=True)