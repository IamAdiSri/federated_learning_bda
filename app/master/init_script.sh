# Use the most recent image and give it Google Cloud API access
# gcloud dataproc clusters create bdacluster --region us-west1 --zone us-west1-b --master-machine-type e2-standard-2 --master-boot-disk-size 50 --num-workers 4 --worker-machine-type e2-standard-2 --worker-boot-disk-size 50 --image-version 2.1-debian11 --scopes 'https://www.googleapis.com/auth/cloud-platform' --tags bda-in,bda-out --initialization-actions 'gs://bdastorage/scripts/init_script.sh' --project big-data-project-384601

$PYSPARK_PYTHON -m pip install flask torch transformers simpletransformers

mkdir /home/dataproc/fl_server
cd /home/dataproc/fl_server

gsutil cp gs://$BUCKET_NAME/scripts/app.py app.py
gsutil cp gs://$BUCKET_NAME/scripts/fl.py fl.py
gsutil cp gs://$BUCKET_NAME/scripts/setup.sh setup.sh

apt-get --assume-yes install tmux

# running this line completely breaks spark on the VMs
# but you can ssh in and run this manually
# sudo $PYSPARK_PYTHON app.py