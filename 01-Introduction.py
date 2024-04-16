# Databricks notebook source
# MAGIC %md
# MAGIC # Biomedical Question Answering over Custom Datasets with 🦜️🔗 LangChain and Open Source LLMs on Hugging Face 🤗 
# MAGIC
# MAGIC Large Language Models produce some amazing results, chatting and answering questions with seeming intelligence. But how can you get LLMs to answer questions about _your_ specific datasets? Imagine answering questions based on your company's knowledge base, docs or Slack chats. The good news is that this is easy with open-source tooling and LLMs. This example shows how to apply [LangChain](https://python.langchain.com/en/latest/index.html), Hugging Face `transformers`, and open source LLMs such as [MPT-7b-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) from MosaicML or [Falcon-7b-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) from the Technology Innovation Institute. This example can make use of any text-generation LLM or even OpenAI with minor changes. In this case, the data set is a set of freely available published papers in PDF format about cystic fibrosis from PubMed, but could be any corpus of text.

# COMMAND ----------

# MAGIC %md
# MAGIC <img style="margin: auto; display: block" width="1200px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/solution-overview.jpeg?token=GHSAT0AAAAAACBNXSB43DCNDGVDBWWEWHZCZEBLWBA">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Ingestion and Preparation
# MAGIC
# MAGIC <img style="float: right" width="500px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/data-prep.jpeg?token=GHSAT0AAAAAACBNXSB4IK2XJS37QU6HCJCEZEBL3TA">
# MAGIC
# MAGIC
# MAGIC #
# MAGIC 1. Organize your documents into a directory on DBFS or S3 (DBFS is easier but S3 works too)
# MAGIC     * In this demo we have preuploaded a set of PDFs from PubMed on S3, but your own documents will work the same way
# MAGIC 2. Use LangChain to ingest those documents and split them into manageable chunks using a text splitter
# MAGIC 3. Use a sentence transformer NLP model to create embeddings of those text chunks and store them in a vectorstore
# MAGIC     * Embeddings are basically creating a high-dimension vector encoding the semantic meaning of a chunk of text
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Chain Creation
# MAGIC
# MAGIC <img style="float: right" width="500px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/llm-chain.jpeg?token=GHSAT0AAAAAACBNXSB4UGOIIYZJ37LBI4MOZEBL4LQ">
# MAGIC
# MAGIC #
# MAGIC Construct a chain using LangChain such that when a user submits a question to the chain the following steps happen:
# MAGIC 1. Similarity search for your question on the vectorstore, i.e. “which chunks of text have similar context/meaning as the question?”
# MAGIC 2. Retrieve the top `k` chunks
# MAGIC 3. Submit relevant chunks and your original question together to the LLM
# MAGIC 4. LLM answers the question with the relevant chunks as a reference
# MAGIC
# MAGIC We will also need to define some critical parameters, such as which LLM to use, how many text chunks (`k`) to retrieve, and model performance parameters.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Cluster Setup
# MAGIC
# MAGIC - Run the next notebooks on a cluster with Databricks Runtime 13.0 ML GPU. It should work on 12.2 ML GPU as well.
# MAGIC - To run this notebook's examples all that is needed is a single-node 'cluster', with a single A10 GPU (ex: `g5.4xlarge` on AWS). A100 instances work as well.
# MAGIC
# MAGIC We have provided code to create a sample cluster for this notebook. See the `./RUNME` notebook for instructions on how to use this automation.
