# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/hls-llm-doc-qa

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Ingestion and Preparation
# MAGIC
# MAGIC <img style="float: right" width="800px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/data-prep.jpeg?token=GHSAT0AAAAAACBNXSB4IK2XJS37QU6HCJCEZEBL3TA">
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
# MAGIC Start with required Python libraries for data preparation.

# COMMAND ----------

# MAGIC %run ./util/install-prep-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

# where you want the PDFs to be saved in your environment
dbutils.widgets.text("PDF_Path", "/dbfs/tmp/langchain_hls/pdfs")

# which embeddings model from Hugging Face 🤗  you would like to use; for biomedical applications we have been using this model recently
# also worth trying this model for embeddings for comparison: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
dbutils.widgets.text("Embeddings_Model", "pritamdeka/S-PubMedBert-MS-MARCO")

# where you want the vectorstore to be persisted across sessions, so that you don't have to regenerate
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/langchain_hls/db")

# publicly accessible bucket with PDFs for this demo
dbutils.widgets.text("Source_Documents", "s3a://db-gtm-industry-solutions/data/hls/llm_qa/")


# Vector Search Endpoint Name 
dbutils.widgets.text("Vector_Search_Endpoint", "hls_llm_qa_demo_vse")

# Vector Index Name 
dbutils.widgets.text("Vector_Index", "hls_llm_qa_demo_ws.vse.hls_llm_qa_hf_embeddings")


# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

#get widget values
pdf_path = dbutils.widgets.get("PDF_Path")
source_pdfs = dbutils.widgets.get("Source_Documents")
db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")
embeddings_model = dbutils.widgets.get("Embeddings_Model")
vector_search_endpoint_name = dbutils.widgets.get("Vector_Search_Endpoint")
vector_index_name = dbutils.widgets.get("Vector_Index")

# COMMAND ----------

import os
# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Prep
# MAGIC
# MAGIC This data preparation need only happen one time to create data sets that can then be reused in later sections without re-running this part.
# MAGIC
# MAGIC - Grab the set of PDFs (ex: Arxiv papers allow curl, PubMed does not)
# MAGIC - We have are providing a set of PDFs from PubMedCentral relating to Cystic Fibrosis (all from [PubMedCentral Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/), all with the CC BY license), but any topic area would work
# MAGIC - If you already have a repository of PDFs then you can skip this step, just organize them all in an accessible DBFS location

# COMMAND ----------

import os
import shutil

# in case you rerun this notebook, this deletes the directory and recreates it to prevent file duplication
if os.path.exists(pdf_path):
  shutil.rmtree(pdf_path, ignore_errors=True)
os.makedirs(pdf_path)

# slightly modifying the file path from above to work with the dbutils.fs syntax
modified_pdf_path = "dbfs:/" + pdf_path.lstrip("/dbfs")
dbutils.fs.cp(source_pdfs, modified_pdf_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC All of the PDFs should now be accessible in the `pdf_path` now; you can run the below command to check if you want.
# MAGIC
# MAGIC `!ls {pdf_path}`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Document DB
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. 
# MAGIC These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Use `langchain` to reading directly from PDFs, although LangChain also supports txt, HTML, Word docs, GDrive, PDFs, etc.
# MAGIC - Create a simple in-memory Chroma vector DB for storage
# MAGIC - Instantiate an embedding function from `sentence-transformers`
# MAGIC - Populate the database and save it

# COMMAND ----------

# MAGIC %md
# MAGIC Prepare a directory to store the document database. Any path on `/dbfs` will do.

# COMMAND ----------

!(rm -r {db_persist_path} || true) && mkdir -p {db_persist_path}

# COMMAND ----------

# MAGIC %md
# MAGIC Create the document database:
# MAGIC - Here we are using the `PyPDFDirectoryLoader` loader from LangChain ([docs page](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pypdf)) to form `documents`; `langchain` can also form doc collections directly from PDFs, GDrive files, etc.

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader

loader_path = f"{pdf_path}/"

pdf_loader = PyPDFDirectoryLoader(loader_path)
docs = pdf_loader.load()
len(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are using a text splitter from LangChain to split our PDFs into manageable chunks. This is for a few reasons, primarily:
# MAGIC - LLMs (currently) have a limited context length. MPT-7b-Instruct by default can only accept 2048 tokens (roughly words) in the prompt, although it can accept 4096 with a small settings change. This is rapidly changing, though, so keep an eye on it.
# MAGIC - When we create embeddings for these documents, an NLP model (sentence transformer) creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - In this case, the embeddings model we use can except a very limited number of tokens. The default one we have selected in this notebook, [
# MAGIC S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO), has also been finetuned on a PubMed dataset, so it is particularly good at generating embeddings for medical documents.
# MAGIC - More info on embeddings: [Hugging Face: Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)

# COMMAND ----------

# For PDFs we need to split them for embedding:
from langchain.text_splitter import TokenTextSplitter

# this is splitting into chunks based on a fixed number of tokens
# the embeddings model we use below can take a maximum of 128 tokens (and truncates beyond that) so we keep our chunks at that max size
text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=32)
documents = text_splitter.split_documents(docs)

# COMMAND ----------

display(documents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we split the documents into more manageable chunks. We will now set up **Databricks Vector Search** with a **Direct Vector Access Index** which will be used with Langchain in our RAG architecture. 
# MAGIC - We first need to create a dataframe with an id column to be used with Vector Search.
# MAGIC - We will then calculate the embeddings using HuggingFaceEmbeddgins
# MAGIC - Finally we will save this in our Vector Search as an index to be used for RAG.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Automatically generates a PAT Token for authentication
client = VectorSearchClient()

# Uses the service principal token for authentication
# client = VectorSearch(service_principal_client_id=<CLIENT_ID>,service_principal_client_secret=<CLIENT_SECRET>)

# client.create_endpoint(
#     name= vector_search_endpoint_name,
#     endpoint_type="STANDARD"
# )

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id

# add id for the primary key for the vector search index. Also cast metadata to string instead of map<string, string> which is incompatible with vector search 
documents_with_id = spark.createDataFrame(documents).withColumn("metadata", col("metadata").cast("string")).withColumn("id", monotonically_increasing_id())

# Write the dataframe to Unity Catalog to be used as source table
# documents_with_id.write.option("mergeSchema", "true").mode("overwrite").format("delta").saveAsTable("hls_llm_qa_demo_ws.vse.hls_llm_qa_raw_docs")

# display(documents_with_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE hls_llm_qa_demo_ws.vse.hls_llm_qa_raw_docs SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #TODO 
# MAGIC
# MAGIC - Confirm embedding dimension
# MAGIC - index requires a UC target to save 
# MAGIC - 

# COMMAND ----------

index = client.create_delta_sync_index(
  endpoint_name= vector_search_endpoint_name,
  source_table_name= "hls_llm_qa_demo_ws.vse.hls_llm_qa_raw_docs",
  index_name= vector_index_name,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column= "page_content",
  embedding_model_endpoint_name="databricks-bge-large-en" 
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
# MAGIC - Load the Chroma DB
# MAGIC - Instantiate an LLM, like Dolly here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

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
vectorstore = get_retriever()

similar_documents = vectorstore.get_relevant_documents("What is cystic fibrosis?")
print(f"Relevant documents: {similar_documents[0]}")
