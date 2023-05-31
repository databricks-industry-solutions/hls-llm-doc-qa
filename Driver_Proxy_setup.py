# Databricks notebook source
# MAGIC %run ./util/install-llm-libraries

# COMMAND ----------

import os

# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"
HF_HOME = "/dbfs/tmp/cache/hf"

# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
os.environ['HF_HOME'] = HF_HOME

# COMMAND ----------

dbutils.widgets.text('model_name','mosaicml/mpt-7b-instruct')
#dbutils.widgets.text('input_max_seq_len','2048')
dbutils.widgets.text('num_beams','1')
dbutils.widgets.text('max_new_tokens','128')

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
num_beams = dbutils.widgets.get("num_beams")
max_new_tokens = dbutils.widgets.get("max_new_tokens")
#input_max_seq_len = dbutils.widgets.get("input_max_seq_len")

# COMMAND ----------

import torch
import transformers

config = transformers.AutoConfig.from_pretrained(
  model_name,
  trust_remote_code=True,
  max_new_tokens=max_new_tokens
)
config.attn_config['attn_impl'] = 'triton'

config.update({"max_seq_len": 4096})

model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True
)

# COMMAND ----------

device = 'cuda'

# COMMAND ----------

model.to(device)

# COMMAND ----------

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# COMMAND ----------

# See why we need this format: http://fastml.com/how-to-train-your-own-chatgpt-alpaca-style-part-one/
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}
 
### Response:
"""

# COMMAND ----------

import time
import logging

def gen_text(prompt, stop=["#"], **kwargs):
  full_prompt = prompt_template.format(prompt=prompt)
  inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
  if 'max_new_tokens' not in kwargs:
      kwargs['max_new_tokens'] = 256
  if 'temperature' not in kwargs:
      kwargs['temperature'] = 0.0
  if stop and 'eos_token_id' not in kwargs:
      eos_token_id = [tokenizer.encode(s, add_special_tokens=False) for s in stop]
      kwargs['eos_token_id'] = eos_token_id
      kwargs['pad_token_id'] = tokenizer.eos_token_id
  start = time.time()
  outputs = model.generate(inputs, **kwargs)
  duration = time.time() - start
  n_tokens = len(outputs[0, ])
  logging.warning(f"{n_tokens/duration} tokens/sec, {n_tokens} tokens")
  result = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
  return result

# COMMAND ----------

eos_token_id = [tokenizer.encode(s, add_special_tokens=False) for s in ["#"]]

# COMMAND ----------

from flask import Flask, request, jsonify

# COMMAND ----------

app = Flask("mpt-7b-instruct")

@app.route('/', methods=['POST'])
def serve_gen_text():
  resp = gen_text(**request.json)
  return jsonify(resp)

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(driver_proxy_api)

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------

# MAGIC %md Client code
# MAGIC
# MAGIC See `driver_proxy_api` value printed above.
# MAGIC
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC def gen_text(prompt):
# MAGIC     resp = requests.post(driver_proxy_api, json={"prompt": prompt}, headers={"Authorization": f"Bearer {api_token}"})
# MAGIC     if resp.ok:
# MAGIC         return resp.text
# MAGIC     raise
# MAGIC ```
