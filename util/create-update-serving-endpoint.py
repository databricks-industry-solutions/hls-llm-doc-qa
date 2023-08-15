# Databricks notebook source
import mlflow
import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds


# gather other inputs the API needs - they are used as environment variables in the
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

def endpoint_exists(serving_endpoint_name):
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint(serving_endpoint_name):
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint(serving_endpoint_name, served_models):
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {serving_endpoint_name}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": serving_endpoint_name, "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")
  
def update_endpoint(serving_endpoint_name, served_models):
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {serving_endpoint_name}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")
