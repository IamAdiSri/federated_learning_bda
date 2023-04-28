# Use the most recent image and give it Google Cloud API access
# gcloud dataproc clusters create bdacluster --region us-west1 --zone us-west1-b --master-machine-type e2-standard-2 --master-boot-disk-size 50 --num-workers 4 --worker-machine-type e2-standard-2 --worker-boot-disk-size 50 --image-version 2.1-debian11 --scopes 'https://www.googleapis.com/auth/cloud-platform' --initialization-actions 'gs://bdastorage/scripts/init_script.sh' --project big-data-project-384601

$PYSPARK_PYTHON -m pip install flask torch transformers simpletransformers

export BUCKET_NAME="bdastorage"
export AVG_ALGO="mean"
export MODEL_VER=0

mkdir fl_server
cd fl_server

gsutil cp gs://$BUCKET_NAME/scripts/app.py app.py
gsutil cp gs://$BUCKET_NAME/scripts/fl.py fl.py

apt-get --assume-yes install tmux

# running this line completely breaks spark on the VMs
# $PYSPARK_PYTHON app.py