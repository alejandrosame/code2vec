# coding: utf-8

from sourced.engine import Engine
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder\
        .master("local[*]")\
        .appName("Examples")\
        .getOrCreate()

engine = Engine(spark, "/repositories/test", "siva")

print("%d repositories successfully loaded" % (engine.repositories.count()))

seed=1234


#  ## Get branch with latest commit for each repository

df = engine.repositories.references\
     .filter("is_remote = true")\
     .filter("NOT name LIKE 'refs/heads/HEAD' AND name LIKE 'refs/heads/%'")\
     .commits\
     .cache()
    
latest_commit = df.groupBy("repository_id").agg(max("committer_date").alias("date"))\
                  .withColumnRenamed('repository_id', 'id')

latest_repo_ref = df.select("repository_id", "reference_name", "committer_date")\
                    .join(latest_commit, 
                          (df.repository_id == latest_commit.id) 
                           & (df.committer_date == latest_commit.date))\
                    .select('repository_id', 'reference_name')\
                    .cache()                 

print("First 10 repo branches:")
latest_repo_ref.show(10)

print("Repository count after join %d " % (latest_repo_ref.count()))


# ## Splitting train and test sets
# 
# We can use randomSplit over the repositories DataFrame to get the train and test sets. 
# The same can be done later to get the train and validation sets depending on the specific 
# cross-validation approach used. 

data = latest_repo_ref\
       .withColumnRenamed('repository_id', 'repo_id')\
       .withColumnRenamed('reference_name', 'ref_name')

[train, test] = data.randomSplit([0.8, 0.2], seed)

print("Total count %d || Train count %d || Test count %d" % (data.count(), 
                                                             train.count(),
                                                             test.count()))


# ## Prepare train dataset with Python UASTs

# First, get Python blobs with UASTs:

# Get repo_ids and ref_names to filter and avoid extracting UASTs on all blobs
repo_ids, ref_names = set(), set()

select = latest_repo_ref.select("repository_id", "reference_name").collect()
for row in select: repo_ids.add(row.repository_id); ref_names.add(row.reference_name);

# Get blobs with UASTs
python_blobs = df.blobs\
               .repartition(32)\
               .filter(df.blobs.repository_id.isin(repo_ids))\
               .filter(df.blobs.reference_name.isin(ref_names))\
               .classify_languages()\
               .filter("is_binary = false")\
               .filter("lang = 'Python'")\
               .dropDuplicates(['blob_id'])    .cache()

python_blobs.count()

python_uasts = python_blobs\
               .repartition(32)\
               .extract_uasts()\
               .drop("content")\
               .cache()

python_uasts.count()


# Join python_blobs dataframe with the train dataframe to get UASTS we want to use for training:

train_python = train\
    .join(python_uasts, 
       (train.repo_id == python_uasts.repository_id) 
        & (train.ref_name == python_uasts.reference_name))\
    .cache()


train_python.count()

train_python.printSchema()

