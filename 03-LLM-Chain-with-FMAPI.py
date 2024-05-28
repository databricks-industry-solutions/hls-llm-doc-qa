# Databricks notebook source
# MAGIC %md
# MAGIC ## Using an LLM Served on Databricks Model Serving: A LangChain app
# MAGIC
# MAGIC <img style="float: right" width="800px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/llm-chain.jpeg?token=GHSAT0AAAAAACBNXSB4UGOIIYZJ37LBI4MOZEBL4LQ">
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
# MAGIC Start with required libraries.

# COMMAND ----------

# MAGIC %run ./util/install-langchain-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection from the previous step, as well as defining where our vectorstore was persisted and which embeddings model we want to use.

# COMMAND ----------


# which embeddings model we want to use. We are going to use the foundation model API, but you can use custom models (i.e. from HuggingFace), External Models (Azure OpenAI), etc.
dbutils.widgets.text("Embeddings_Model", "databricks-bge-large-en")

# which LLM model we want to use. We are going to use the foundation model API, but you can use custom models (i.e. from HuggingFace), External Models (Azure OpenAI), etc.
dbutils.widgets.text("FMAPI_Model", "databricks-dbrx-instruct")

# Location for the split documents to be saved  
dbutils.widgets.text("Persisted_UC_Table_Location", "hls_llm_qa_demo.vse.hls_llm_qa_raw_docs")

# Vector Search Endpoint Name - one-env-shared-endpoint-7, hls_llm_qa_demo_vse
dbutils.widgets.text("Vector_Search_Endpoint", "one-env-shared-endpoint-7")

# Vector Index Name 
dbutils.widgets.text("Vector_Index", "hls_llm_qa_demo.vse.hls_llm_qa_embeddings")

# COMMAND ----------

#get widget values
model_endpoint_name= dbutils.widgets.get("FMAPI_Model")

# dbutils.widgets.get('model_name_from_model_serving')
embeddings_model = dbutils.widgets.get("Embeddings_Model")

vector_search_endpoint_name = dbutils.widgets.get("Vector_Search_Endpoint")
vector_index_name = dbutils.widgets.get("Vector_Index")
UC_table_save_location = dbutils.widgets.get("Persisted_UC_Table_Location")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Load Databricks Vector Search and define our retriever. We define `k` here, which is how many chunks of text we want to retrieve from the vectorstore to feed into the LLM
# MAGIC - Instantiate an LLM, loading from Databricks Model serving here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

vs_index = vsc.get_index(endpoint_name= vector_search_endpoint_name, index_name= vector_index_name)

vs_index.describe()

# COMMAND ----------

from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


def get_retriever(persist_dir: str = None):
    
    vs_index = vsc.get_index(
        endpoint_name= vector_search_endpoint_name,
        index_name= vector_index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="page_content"
    )
    return vectorstore.as_retriever()


# test our retriever
retriever = get_retriever()

# COMMAND ----------

# If running a Databricks notebook attached to an interactive cluster in "single user"
# or "no isolation shared" mode, you only need to specify the endpoint name to create
# a `Databricks` instance to query a serving endpoint in the same workspace.

# Otherwise, you can manually specify the Databricks workspace hostname and personal access token
# or set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, respectively.
# You can set those environment variables based on the notebook context if run on Databricks

import os
from langchain_community.llms import Databricks

# Need this for job run: 
os.environ['DATABRICKS_URL'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

from langchain.llms import Databricks
from langchain_core.messages import HumanMessage, SystemMessage

def transform_input(**request):
  request["messages"] = [
    {
      "role": "user",
      "content": request["prompt"]
    }
  ]
  del request["prompt"]
  return request

llm = Databricks(endpoint_name="databricks-dbrx-instruct", transform_input_fn=transform_input)

#if you want answers to generate faster, set the number of tokens above to a smaller number
prompt = "What is cystic fibrosis?"

displayHTML(llm(prompt))

# COMMAND ----------

# DBTITLE 1,Build the QA Chain using Vector Search as the retreiver
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain():
  
  template = """You are a life sciences researcher with deep expertise in cystic fibrosis and related comorbidities. Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  Use only information in the following paragraphs to answer the question. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
  
  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever= retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt
    }
  )
  
  # Set verbose=True to see the full prompt:
  return qa_chain

# COMMAND ----------

# MAGIC %md
# MAGIC Note that there are _many_ factors that affect how the language model answers a question. Most notable is the prompt template itself. This can be changed, and different prompts may work better or worse with certain models.
# MAGIC
# MAGIC The generation process itself also has many knobs to tune, and often it simply requires trial and error to find settings that work best for certain models and certain data sets. See this [excellent guide from Hugging Face](https://huggingface.co/blog/how-to-generate). 
# MAGIC
# MAGIC The settings that most affect performance are:
# MAGIC - `max_new_tokens`: longer responses take longer to generate. Reduce for shorter, faster responses
# MAGIC - `k`: the number of chunks of text retrieved into the prompt. Longer prompts take longer to process
# MAGIC - `num_beams`: if using beam search, more beams increase run time more or less linearly

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the Chain for Simple Question Answering
# MAGIC
# MAGIC That's it! it's ready to go. Define a function to answer a question and pretty-print the answer, with sources.
# MAGIC
# MAGIC 🚨 Note:
# MAGIC Here we are using an LLM without any fine tuning on a specialized dataset. As a result, similar to any other question/answering model, the LLM's results are not reliable and can be factually incorrect.

# COMMAND ----------



# COMMAND ----------

qa_chain = build_qa_chain()

# COMMAND ----------

question = "Is probability a class topic?"
result = qa_chain({"query": question})
# Check the result of the query
result["result"]
# Check the source document from where we 
result["source_documents"][0]

# COMMAND ----------

def answer_question(question):
  qa_chain = build_qa_chain()
  result = qa_chain({"query": question})
  answer = result["result"]
  source_docs = result["source_documents"]
  displayHTML(answer)
  displayHTML(source_docs)

# COMMAND ----------

# MAGIC %md 
# MAGIC Try asking a question about cystic fibrosis!

# COMMAND ----------

answer_question("What are the primary drugs for treating cystic fibrosis (CF)?")

# COMMAND ----------

answer_question("What are the cystic fibrosis drugs that target the CFTR protein?")

# COMMAND ----------


