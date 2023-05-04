import io
import zipfile
import requests
import os
import time
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import logging

host = "34.102.116.94"
port = "5000"

class ClientTrain:

    def __init__(self, host, port):
        """
        1. Maintain a unique ID for each copy operation
        """
        self.version = 0
        self.model = None
        self.endpoint = f"http://{host}:{port}"
        # self.extract_path = "path/to/extract"
        self.model_path = "path/of/trained/model"
        self.wait_time = 1000
        self.sample_size = 0

    def get_latest_model(self):
        model_fetch_url = f"{self.endpoint}/fetch_model/{self.version}"
        response = requests.get(model_fetch_url)
    
        if response.status == 200:
            zip_bytes = response.response
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
                # Extract the files from the zip file
                zip_file.extractall(path = self.model_path)
            print('File downloaded and extracted successfully')
            return True

        elif response.status == 404:
            server_model_version = response.server_version
            if self.version > server_model_version:
                print("Server not ready with this model")
                time.sleep(self.wait_time)
                self.get_latest_model()
            else:
                self.version = server_model_version
                print(f'Newer Model Available on server')
                self.get_latest_model()

        elif response.status == 500:
            print("Server Issue. Retrying ...")
            time.sleep(self.wait_time)
            self.get_latest_model()
        
        else:
            print(f"ERROR receiving file {response}")
            time.sleep(self.wait_time)
            self.get_latest_model()


    def send_trained_model(self):
        model_push_url =  f"{self.endpoint}/upload_model/{self.version}"
        # Create a zip file object in memory
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            # Add all the files in the folder to the zip file
            for root, dirs, files in os.walk(self.model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path)

        # Send the zip file to the server
        response = requests.post(model_push_url, files=zip_bytes.getvalue(), data = {"sample_size":self.sample_size})
        if response.status == 200:
            print('Zip file sent successfully')
            return True
        elif response.status == 404:
            if self.version != response.server_version:
                print("Discarding current model")
                return False
            else:
                print("Waiting for server to reach current version or finish processing")
                time.sleep(self.wait_time)
                self.send_trained_model()
        else:
            print(f'ERROR sending zip file {response}')
            return False

    def train_model(self):
        
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        # Preparing train data
        train_df = pd.read_csv("/Users/harsh/Desktop/CU/BigData/project/bda_project_spring_23/datasets/train.csv")
        train_df.columns = ["text", "labels"]

        self.sample_size = train_df
        # Optional model configuration
        model_args = ClassificationArgs(num_train_epochs=1)

        # Load the model
        self.model = ClassificationModel("distilbert", self.model_path, args=model_args, use_cuda=False)

        # Train the model
        self.model.train_model(train_df)

        # Delete the old model from the model path
        os.system(f'rm -rf {self.model_path}/*')

        # Save the model in the model_path
        self.model.save_model_args(self.model_path)
        self.model.tokenizer.save_pretrained(self.model_path)
        self.model.model.save_pretrained(self.model_path)

client = ClientTrain(host, port)
while True:
    if client.get_latest_model():
        client.train_model()
        client.send_trained_model() 
        # would keep trying to send and returns after:
        # 1. successfuly sending, 2. Server is ahead, 3. Unknown error

        # Delete the existing model from model path
        os.system(f'rm -rf {client.model_path}/*')
    else:
        # Delete any junk present in the model path
        os.system(f'rm -rf {client.model_path}/*')
        time.sleep(client.wait_time)

