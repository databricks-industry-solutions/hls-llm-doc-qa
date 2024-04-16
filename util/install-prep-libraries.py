# Databricks notebook source
# Potentially used: unstructured[local-inference] 

# COMMAND ----------

# MAGIC %pip install -U transformers==4.39.3 sentence-transformers==2.6.1 mlflow==2.11.3 langchain==0.1.16 databricks-vectorsearch==0.28 pypdf==4.2.0 pycryptodome==3.20.0 accelerate==0.29.2 sacremoses==0.1.1 ninja==1.11.1.1 tiktoken==0.6.0 nltk==3.7
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')
