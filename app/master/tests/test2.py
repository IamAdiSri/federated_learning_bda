# checks if pyspark can access bucket
import time
from google.cloud import storage
from pyspark import SparkConf
from pyspark.sql import SparkSession

sparkConf = SparkConf()
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

def mapp(addr, s):
    # client objects cannot be defined outside the scope of the function
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("bdastorage")
    blob = bucket.get_blob(addr)
    return blob.size/s

def reduce(a, b):
    return a+b

ct = time.time()
s = 1000 # takes around 33 seconds
addr = "models/global_v0.zip"
rdd = spark.sparkContext.parallelize([addr for _ in range(s)])
soln = rdd.map(lambda x: mapp(addr, s)).reduce(reduce)

print(soln, time.time()-ct)

