# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/hls-llm-doc-qa

# COMMAND ----------

# MAGIC %md
# MAGIC # Manage Llama 2 7B chat model with MLFlow on Databricks
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS
# MAGIC
# MAGIC GPU instances that have at least 16GB GPU memory would be enough for inference on single input (batch inference requires slightly more memory). On Azure, it is possible to use `Standard_NC6s_v3` or `Standard_NC4as_T4_v3`.
# MAGIC
# MAGIC requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve model info from the previous step

# COMMAND ----------

#this should be the name of the registered model from the previous step
model_name = 'Llama-2-7b-MedText-QLoRa'

# COMMAND ----------

# Initialize the Model Registry client
import mlflow
from mlflow.tracking.client import MlflowClient
client = MlflowClient

def get_latest_model_version(model_name: str):
  client = MlflowClient()
  models = client.get_latest_versions(model_name, stages=["None"])
  for m in models:
    new_model_version = m.version
  return new_model_version

  model_version = get_latest_model_version(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint from Registered Model
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the MPT-7B-Instruct model.

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'llama2-7b-MedText-QLoRA'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "model_name": model_name,
      "model_version": model_version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `05-LLM-Chain-with-GPU-Serving` for example code) running in the same workspace.
