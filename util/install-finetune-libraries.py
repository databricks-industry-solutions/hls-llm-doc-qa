# Databricks notebook source
# MAGIC %pip install -U git+https://github.com/huggingface/transformers.git  git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git
# MAGIC %pip install datasets==2.12.0 bitsandbytes==0.40.1 einops==0.6.1 trl==0.4.7
# MAGIC %pip install torch==2.0.1
# MAGIC %pip install safetensors

# COMMAND ----------

dbutils.library.restartPython()
