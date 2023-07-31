# Databricks notebook source
# MAGIC %md
# MAGIC We are installing some NVIDIA libraries here in order to use a special package called Flash Attention (`flash_attn`) that Mosaic ML's mpt-7b-instruct model needs. This can take 5-10 minutes

# COMMAND ----------

# MAGIC %pip install -U transformers==4.29.2 sentence-transformers==2.2.2 langchain==0.0.190 chromadb==0.3.25 pypdf==3.9.1 pycryptodome==3.18.0 accelerate==0.19.0 unstructured==0.7.1 unstructured[local-inference]==0.7.1 sacremoses==0.0.53 ninja==1.11.1 torch==2.0.1 pytorch-lightning==2.0.1 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#only run this step if you are running below Databricks runtime 13.2 GPU; if using 13.2 and above this is not needed.
%sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install -y libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7

# COMMAND ----------

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

import nltk
nltk.download('punkt')
