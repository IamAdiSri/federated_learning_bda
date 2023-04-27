# tests pyspark by calculating mean of 0 to 9999
from pyspark import SparkConf
from pyspark.sql import SparkSession

sparkConf = SparkConf()
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

def mapp(n, s):
    return n/s

def reduce(a, b):
    return a+b

s = 10000
rdd = spark.sparkContext.parallelize(list(range(s)))
soln = rdd.map(lambda x: mapp(x, s)).reduce(reduce)

print(soln)
