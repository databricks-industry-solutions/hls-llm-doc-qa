# Databricks notebook source
# MAGIC %pip install -U transformers langchain chromadb pypdf pycryptodome accelerate unstructured unstructured[local-inference] sacremoses ninja nltk

# COMMAND ----------

# MAGIC %md
# MAGIC We are installing some NVIDIA libraries here in order to use a special package called Flash Attention (`flash_attn`) that Mosaic ML's mpt-7b-instruct model needs.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# MAGIC sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# MAGIC sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# MAGIC sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# MAGIC sudo apt-get update

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install -y libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7

# COMMAND ----------

# MAGIC %md
# MAGIC Installing `flash_attn` takes around 5 minutes.

# COMMAND ----------

# MAGIC %pip install einops flash_attn

# COMMAND ----------

import nltk
nltk.download('punkt')
