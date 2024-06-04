# Databricks notebook source
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
# MAGIC 3. Use a BAAI general embedding model to create embeddings of those text chunks and store them in a Databricks Vector Search index
# MAGIC     * Embeddings are basically creating a high-dimension vector encoding the semantic meaning of a chunk of text
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Start with required Python libraries for data preparation.

# COMMAND ----------

# MAGIC %run ./util/install-prep-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

# Target catalog name
dbutils.widgets.text("catalog_name", "hls_llm_qa_demo")
# Target vector search schema name
dbutils.widgets.text("vector_search_schema_name", "hls_llm_vse")

catalog_name = dbutils.widgets.get("catalog_name")
vector_search_schema_name = dbutils.widgets.get("vector_search_schema_name")

# COMMAND ----------

# Where you want the PDFs to be saved in your environment
dbutils.widgets.text("uc_volume_path", f"{catalog_name}.data.pdf_docs")

# Which embeddings model we want to use. We are going to use the foundation model API, but you can use custom models (i.e. from HuggingFace), External Models (Azure OpenAI), etc.
dbutils.widgets.text("embedding_model_name", "databricks-bge-large-en")

# Publicly accessible bucket with PDFs for this demo
dbutils.widgets.text("source_documents", "s3a://db-gtm-industry-solutions/data/hls/llm_qa/")

# Location for the split documents to be saved  
dbutils.widgets.text("persisted_uc_table_path", f"{catalog_name}.{vector_search_schema_name}.hls_llm_qa_raw_docs")

# Vector Search endpoint name
dbutils.widgets.text("vector_search_endpoint_name", "hls_llm_qa_vse")

# Vector index name 
dbutils.widgets.text("vector_index", f"{catalog_name}.{vector_search_schema_name}.hls_llm_qa_embeddings")

# COMMAND ----------

#get widget values and store in Python variables

uc_volume_path = dbutils.widgets.get("uc_volume_path")
source_pdfs = dbutils.widgets.get("source_documents")
embeddings_model = dbutils.widgets.get("embedding_model_name")
vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")
vector_index_name = dbutils.widgets.get("vector_index")
UC_table_save_location = dbutils.widgets.get("persisted_uc_table_path")

# Volume path string uses slashes with the saved uc_volume_path
target_volume_path = f"/Volumes/{catalog_name}/data/pdf_docs"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create Unity catalog if it does not exist
# MAGIC -- Use IF NOT EXISTS clause to avoid errors if the catalog already exists
# MAGIC CREATE CATALOG IF NOT EXISTS ${catalog_name}

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create Unity schema if it does not exist in the Unity catalog
# MAGIC -- Use IF NOT EXISTS clause to avoid errors if the schema already exists
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS ${catalog_name}.${vector_search_schema_name};

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Prep
# MAGIC
# MAGIC This data preparation need only happen one time to create data sets that can then be reused in later sections without re-running this part.
# MAGIC
# MAGIC - Grab the set of PDFs (ex: Arxiv papers allow curl, PubMed does not)
# MAGIC - We have are providing a set of PDFs from PubMedCentral relating to Cystic Fibrosis (all from [PubMedCentral Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/), all with the CC BY license), but any topic area would work
# MAGIC - If you already have a repository of PDFs then you can skip this step, just organize them all in an accessible Unity Catalog Volume

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an external volume under the newly created directory
# MAGIC CREATE SCHEMA IF NOT EXISTS ${catalog_name}.data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an external volume under the newly created directory
# MAGIC CREATE VOLUME IF NOT EXISTS ${uc_volume_path}
# MAGIC  COMMENT 'This is the managed volume for the PDF documents'

# COMMAND ----------

import os
import shutil

# Copy the files from S3 to the Unity Catalog Volumes  (s3a://db-gtm-industry-solutions/data/hls/llm_qa/)
dbutils.fs.cp(source_pdfs, target_volume_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC All of the PDFs should now be accessible in the `Unity Catalog Volume` now; you can run the below command to check if you want.
# MAGIC
# MAGIC `dbutils.fs.ls(volume_path)`

# COMMAND ----------

dbutils.fs.ls(target_volume_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Document DB
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. 
# MAGIC These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Use `langchain` to reading directly from PDFs, although LangChain also supports txt, HTML, Word docs, GDrive, PDFs, etc.
# MAGIC - Create a Databricks Vector Search endpoint to have a persistent vector index.
# MAGIC - Use the Foundation Model APIs to generate the embeddings to sync against the vector index.
# MAGIC - Sync the vector index to populate for our RAG implementation.

# COMMAND ----------

# MAGIC %md
# MAGIC Create the document database:
# MAGIC - Here we are using the `PyPDFDirectoryLoader` loader from LangChain ([docs page](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pypdf)) to form `documents`; `langchain` can also form doc collections directly from PDFs, GDrive files, etc.

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader

# Load directly from Unity Catalog Volumes
loader_path = target_volume_path

pdf_loader = PyPDFDirectoryLoader(loader_path)
docs = pdf_loader.load()
len(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are using a text splitter from LangChain to split our PDFs into manageable chunks. This is for a few reasons, primarily:
# MAGIC - LLMs (currently) have a limited context length. DRBX by default has a context length of 32k tokens (roughly words) in the prompt.
# MAGIC - When we create embeddings for these documents, an NLP model (databricks-bge-large-en) creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - In this case, the embeddings model we use can accept a limited number of tokens (max length for bge is limited to 512).
# MAGIC - More info on embeddings: [Hugging Face: Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)

# COMMAND ----------

# For PDFs we need to split them for embedding:
from langchain.text_splitter import TokenTextSplitter

# this is splitting into chunks based on a fixed number of tokens
# the embeddings model we use below can take a maximum of 512 tokens (and truncates beyond that) so we keep our chunks at that max size
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=32)
documents = text_splitter.split_documents(docs)

# COMMAND ----------

display(documents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we split the documents into more manageable chunks. We will now set up **Databricks Vector Search** with a **Delta Sync Index** which will be used with Langchain in our RAG architecture. 
# MAGIC - We first need to create a dataframe with an id column to be used with Vector Search.
# MAGIC - We will then calculate the embeddings using databricks-bge-large-en.
# MAGIC - Finally we will save this in our Vector Search as an index to be used for our QA Chain with the VS retriever. 

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id

# add id for the primary key for the vector search index. Also cast metadata to string instead of map<string, string> which is incompatible with vector search 
documents_with_id = spark.createDataFrame(documents).withColumn("metadata", col("metadata").cast("string")).withColumn("id", monotonically_increasing_id())

# Write the dataframe to Unity Catalog to be used as source table
documents_with_id.write.option("mergeSchema", "true").mode("overwrite").format("delta").saveAsTable(UC_table_save_location)

display(documents_with_id)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Automatically generates a PAT Token for authentication
client = VectorSearchClient()

# Uses the service principal token for authentication if needed
# client = VectorSearch(service_principal_client_id=<CLIENT_ID>,service_principal_client_secret=<CLIENT_SECRET>)

# Check if there are endpoints and potential conflicts. 
if client.list_endpoints():
    if vector_search_endpoint_name not in [
        item["name"] for item in client.list_endpoints()["endpoints"]
    ]:
        print("Creating new VSE " + vector_search_endpoint_name)
        client.create_endpoint(
            name=vector_search_endpoint_name, endpoint_type="STANDARD"
        )
    else:
        print(
            "Vector search endpoint: "
            + vector_search_endpoint_name
            + " already exists!"
        )
else:
    print("Creating new VSE " + vector_search_endpoint_name)
    client.create_endpoint(name=vector_search_endpoint_name, endpoint_type="STANDARD")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Must have Change Data Feed enabled on the UC table in order to use it as the source for an index on Vector Search. 
# MAGIC
# MAGIC ALTER TABLE ${persisted_uc_table_path} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# DBTITLE 1,Check if the endpoint already exists , otherwise create a new one
if client.list_indexes(vector_search_endpoint_name)['vector_indexes']: 
  if vector_index_name not in [item['name'] for item in client.list_indexes(vector_search_endpoint_name)['vector_indexes']]:
    print("Creating vector index: " + vector_index_name)
    index = client.create_delta_sync_index(
    endpoint_name= vector_search_endpoint_name,
    source_table_name= UC_table_save_location,
    index_name= vector_index_name,
    pipeline_type='TRIGGERED',
    primary_key="id",
    embedding_source_column= "page_content",
    embedding_model_endpoint_name= embeddings_model
  )
  else: 
    print("Vector index: " + vector_index_name + " already exists!")
else:
  print("No existing indexes, Creating vector index: " + vector_index_name)
  index = client.create_delta_sync_index(
  endpoint_name= vector_search_endpoint_name,
  source_table_name= UC_table_save_location,
  index_name= vector_index_name,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column= "page_content",
  embedding_model_endpoint_name= embeddings_model
  )

# COMMAND ----------

# DBTITLE 1,Load index using the Vector Search Client
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vs_index = vsc.get_index(endpoint_name= vector_search_endpoint_name, index_name= vector_index_name)

vs_index.describe()

# COMMAND ----------

# DBTITLE 1,Use similarity search to test the index
results = vs_index.similarity_search(
    query_text="What is cystic fibrosis?",
    columns=["id"
             , "page_content"],
    num_results=2
    )

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Load the Vector Search as a retriever.
# MAGIC - Instantiate an LLM, here we use the Foundation Model APIs, but we also use other open source or even OpenAI models.
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt.

# COMMAND ----------

# DBTITLE 1,Use Vector Search Client to create a function to use as retriever for QA chain
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

similar_documents = retriever.get_relevant_documents("What is cystic fibrosis?")

if similar_documents:
    display(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now we are done with the data prep portion and can move on to using an LLM within a full RAG chain! 
