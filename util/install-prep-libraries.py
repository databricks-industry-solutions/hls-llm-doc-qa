# Databricks notebook source
# MAGIC %pip install -U transformers sentence-transformers langchain chromadb pypdf pycryptodome accelerate unstructured unstructured[local-inference] sacremoses ninja tiktoken databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')
