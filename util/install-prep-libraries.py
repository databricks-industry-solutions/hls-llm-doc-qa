# Databricks notebook source
# MAGIC %pip install -U transformers==4.29.2 sentence-transformers==2.2.2 langchain==0.1.4 chromadb==0.3.25 pypdf==3.9.1 pycryptodome==3.18.0 accelerate==0.19.0 unstructured==0.7.1 unstructured[local-inference]==0.7.1 sacremoses==0.0.53 ninja==1.11.1 tiktoken databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')
