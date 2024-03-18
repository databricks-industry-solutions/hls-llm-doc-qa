# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/hls-llm-doc-qa

# COMMAND ----------

# MAGIC %md
# MAGIC # Manage Llama-2-7B-Chat (base model) with MLFlow on Databricks
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

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"
# MAGIC %pip install databricks-sdk==0.12.0 
# MAGIC %pip install safetensors
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import login

# Login to Huggingface to get access to the model if you use the official version of Llama 2
# login(token=dbutils.secrets.get('solution-accelerator-cicd', 'huggingface'))

login(token=dbutils.secrets.get('will_smith_secrets', 'huggingface'))

# COMMAND ----------

import os 
# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("will_smith_secrets", "PAT_HLS")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the model to MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC Download the model

# COMMAND ----------

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main

model_id = "meta-llama/Llama-2-7b-chat-hf" # official version, gated (needs login to Hugging Face)
revision = "01622a9d125d924bd828ab6c72c995d5eda92b8e"

from huggingface_hub import snapshot_download

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
#ignoring .bin files so that we only grab safetensors
snapshot_location = snapshot_download(repo_id=model_id, revision=revision, cache_dir="/local_disk0/.cache/huggingface/", ignore_patterns=["*.bin"])

# COMMAND ----------

# MAGIC %md
# MAGIC Define a customized PythonModel to log into MLFlow.

# COMMAND ----------

import mlflow
import torch
import transformers

# Set mlflow experiment to the user's workspace folder - this enables this notebook to run as part of a job
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/hls-llm-doc-qa'.format(username))

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Define PythonModel to log with mlflow.pyfunc.log_model

class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            use_safetensors=True)
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [1.0])[i]
          max_new_tokens = model_input.get("max_new_tokens", [100])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC Log the model to MLFlow and register the modl using Models in Unity Caalog 

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS ws_models;
# MAGIC
# MAGIC CREATE DATABASE IF NOT EXISTS ws_models.hls_demo

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

mlflow.set_registry_uri("databricks-uc")
model_name = "ws_models.hls_demo.hls_llm_qa_ws"

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_new_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is cystic fibrosis (CF)?"], 
            "temperature": [0.1],
            "max_new_tokens": [75]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 1.7 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        registered_model_name=model_name,
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "safetensors"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# DBTITLE 1,Add Alias to the new model that was registered
from mlflow import MlflowClient
client = MlflowClient()

# create "Champion" alias for version 1 of model "prod.ml_team.iris_model"
client.set_registered_model_alias(model_name, "Champion", 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model from model registry (optional, for testing prior to deployment)
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------


import mlflow
import pandas as pd

# debug
mlflow.set_registry_uri("databricks-uc")
model_name = "ws_models.hls_demo.hls_llm_qa_ws"

model_version_uri = f"models:/{model_name}@Champion"
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "prompt": ["What is ML?", "What is a large language model?"],
        "temperature": [0.1, 0.5],
        "max_new_tokens": [100, 100],
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the MPT-7B-Instruct model.

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve model info from the previous step

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

model_version = get_latest_model_version(model_name)

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

serving_endpoint_name = 'hls_llm_qa_ws_model_endpoint'
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}",  # <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_name} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `04-LLM-Chain-with-GPU-Serving` for example code) running in the same workspace.

# COMMAND ----------


