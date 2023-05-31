# Databricks notebook source
# MAGIC %pip install -U transformers sentence-transformers langchain chromadb pypdf pycryptodome accelerate unstructured unstructured[local-inference] sacremoses ninja nltk

# COMMAND ----------

import nltk
nltk.download('punkt')
