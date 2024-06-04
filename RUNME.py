# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

# MAGIC %md
# MAGIC Before setting up the rest of the accelerator, we need set up a few credentials. Grab a personal access token for your Huggingface account ([documentation](https://huggingface.co/settings/token)). Here we demonstrate using the [Databricks Secret Scope](https://docs.databricks.com/security/secrets/secret-scopes.html) for credential management. 
# MAGIC
# MAGIC Copy the block of code below, replace the name the secret scope and fill in the credentials and execute the block. After executing the code, the accelerator notebook will be able to access the credentials it needs.
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC client = NotebookSolutionCompanion().client
# MAGIC try:
# MAGIC   client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/scopes/create", {"scope": "solution-accelerator-cicd"})
# MAGIC except:
# MAGIC   pass
# MAGIC client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/put", {
# MAGIC   "scope": "solution-accelerator-cicd",
# MAGIC   "key": "huggingface",
# MAGIC   "string_value": "____"
# MAGIC })
# MAGIC ```

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "HLS"
        },
        "tasks": [
            {
                "job_cluster_key": "hls_qa_cluster",
                "notebook_task": {
                    "notebook_path": f"01-Introduction"
                },
                "task_key": "Intro"
            },
            {
                "job_cluster_key": "hls_qa_cluster",
                "notebook_task": {
                    "notebook_path": f"02-Data-prep-with-LangChain"
                },
                "task_key": "Data-prep",
                "depends_on": [
                    {
                        "task_key": "Intro"
                    }
                ]
            },
            {
                "job_cluster_key": "hls_qa_cluster",
                "notebook_task": {
                    "notebook_path": f"03-LLM-Chain-with-FMAPI"
                },
                "task_key": "LLM-chain",
                "depends_on": [
                    {
                        "task_key": "Data-prep"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
              "job_cluster_key": "hls_qa_cluster",
              "new_cluster": {
                  "spark_version": "14.3.x-cpu-ml-scala2.12",
                  "num_workers": 1,
                  "node_type_id": {"AWS": "i3.4xlarge", "MSA": "Standard_D16s_v3", "GCP": "n1-standard-32"}, 
                  "custom_tags": {
                        "usage": "solacc_testing"
                    },
              }
          }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)