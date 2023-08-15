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
# MAGIC %pip install safetensors
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import login

# Login to Huggingface to get access to the model if you use the official version of Llama 2
login(token=dbutils.secrets.get('solution-accelerator-cicd', 'huggingface'))

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
# MAGIC Log the model to MLFlow

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd

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
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "safetensors"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model

# COMMAND ----------

# Register model in MLflow Model Registry
# This may take about 6 minutes to complete
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    name="llama-2-7b-chat",
    await_registration_for=1000,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model from model registry (optional, for testing prior to deployment)
# MAGIC Assume that the below code is run separately or after the memory cache is cleared.
# MAGIC You may need to cleanup the GPU memory.

# COMMAND ----------

"""
import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "prompt": ["What is ML?", "What is large language model?"],
        "temperature": [0.1, 0.5],
        "max_new_tokens": [100, 100],
    }
)

"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the MPT-7B-Instruct model.

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieve model info from the previous step

# COMMAND ----------

#this should be the name of the registered model from the previous step
model_name = 'llama-2-7b-chat'

# Provide a name to the serving endpoint
endpoint_name = 'llama-2-7b-chat'

# COMMAND ----------

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

# MAGIC %run ./util/create-update-serving-endpoint

# COMMAND ----------

served_models = [
    {
      "name": model_name,
      "model_name": model_name,
      "model_version": model_version,
      "workload_size": "Small",
      "workload_type": "GPU_MEDIUM",
      "scale_to_zero_enabled": False
    }
]
traffic_config = {"routes": [{"served_model_name": model_name, "traffic_percentage": "100"}]}

# Create or update model serving endpoint

if not endpoint_exists(endpoint_name):
  create_endpoint(endpoint_name, served_models)
else:
  update_endpoint(endpoint_name, served_models)

# COMMAND ----------

# MAGIC %md 
# MAGIC (Optional) Use the SDK instead of the API Above

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import *
# from datetime import timedelta
# w = WorkspaceClient()
# served_models = [ServedModelInput(model_name=model_name, 
#                                   model_version=model_version, 
#                                   workload_size='Small',
#                                   workload_type='GPU_MEDIUM', # additional param for GPU serving 
#                                   scale_to_zero_enabled='False')]

# try:
#   w.serving_endpoints.create_and_wait(name=endpoint_name, 
#                                      config=EndpointCoreConfigInput(served_models=served_models), 
#                                      timeout=timedelta(minutes=40)) # extending timeout for GPU serving; default is 20
# except: # when the endpoint already exists, update it
#   w.serving_endpoints.update_config_and_wait(name=endpoint_name, 
#                                              served_models=served_models, 
#                                              timeout=timedelta(minutes=40))

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it easily with LangChain (see `04-LLM-Chain-with-GPU-Serving` for example code) running in the same workspace.
