# Databricks notebook source
# MAGIC %md
# MAGIC # Question Answering over Custom Datasets with langchain and Dolly
# MAGIC
# MAGIC <img src="https://www.databricks.com/en-resources-assets/static/97b4d1ab1dbba67b75c57d25c92f1e3a/6c285/lp-heroimage-eb-data-teams-guide-to-lakehouse-platform.png" width=100>
# MAGIC
# MAGIC Large Language Models produce some amazing results, chatting and answering questions with seeming intelligence. But how can you get LLMs to answer questions about _your_ specific datasets? Imagine answering questions based on your company's knowledge base, docs or Slack chats. The good news is that this is easy with open-source tooling and LLMs. This example shows how to apply `langchain`, Hugging Face `transformers`, and even Apache Spark to answer questions about a specific text corpus. It uses the Dolly LLM from Databricks, though this example can make use of any text-generation LLM or even OpenAI with minor changes. In this case, the data set is a set of freely available published papers in PDF format about cystic fibrosis from PubMed, but could be any corpus of text.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Setup
# MAGIC
# MAGIC - Run this on a cluster with Databricks Runtime 13.0 ML GPU. It should work on 12.2 ML GPU as well.
# MAGIC - To run this notebook's examples _without_ distributed Spark inference, all that is needed is a single-node 'cluster', with a single A10 GPU (ex: `g5.4xlarge` on AWS). A100 instances work as well.
# MAGIC
# MAGIC In all events, otherwise start with required Python libraries:

# COMMAND ----------

# MAGIC %pip install -U transformers langchain chromadb pypdf pycryptodome accelerate arxiv unstructured unstructured[local-inference] sacremoses

# COMMAND ----------

# MAGIC %md
# MAGIC We are installing some NVIDIA libraries here in order to use a special package called Flash Attention (`flash_attn`) that Mosaic ML's mpt-7b-instruct model needs.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# MAGIC sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# MAGIC sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# MAGIC sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# MAGIC sudo apt-get update

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install -y libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7

# COMMAND ----------

# MAGIC %pip install ninja

# COMMAND ----------

# MAGIC %md
# MAGIC Installing `flash_attn` takes around 5 minutes.

# COMMAND ----------

# MAGIC %pip install einops flash_attn

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

dbutils.widgets.dropdown('model_name','mosaicml/mpt-7b-instruct',['databricks/dolly-v2-7b','databricks/dolly-v2-3b','mosaicml/mpt-7b-instruct'])

pdf_path = "/dbfs/tmp/langchain_hls/pdfs"
pdf_fs_upload = "/dbfs/FileStore/HLS/langchain/*"
hf_cache_path = "/dbfs/tmp/cache/hf"
db_persist_path = "/dbfs/tmp/langchain_hls/db"

# COMMAND ----------

import os
# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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

!(rm -r {pdf_path} || true) && mkdir -p {pdf_path} && \
  cp -r {pdf_fs_upload} {pdf_path} && \
  ls {pdf_path}

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
# MAGIC - When we create embeddings for these documents, an NLP creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - More info on embeddings: [Hugging Face: Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)

# COMMAND ----------

# For PDFs we need to split them for embedding:
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a small chunk size, but this can change.
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
)
documents = text_splitter.split_documents(docs)

# COMMAND ----------

display(documents)

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

sample_query = "What is cystic fibrosis?"

hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db_persist_path = db_persist_path
db = Chroma.from_documents(collection_name="hls_docs", documents=documents, embedding=hf_embed, persist_directory=db_persist_path)
db.similarity_search(sample_query) 
db.persist()

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

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

db_persist_path = db_persist_path
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(collection_name="hls_docs", embedding_function=hf_embed, persist_directory=db_persist_path)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

def build_qa_chain():
  torch.cuda.empty_cache() # Not sure this is helping in all cases, but can free up a little GPU mem
  model_name=dbutils.widgets.get('model_name') #selected from the dropdown widget at the top of the notebook
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") #for use with mpt-7b

  model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True) #for use with mpt-7b

  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  Use only information in the following paragraphs to answer the question. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
  
  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  end_key_token_id = tokenizer.encode("### End")[0]

  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, \
    pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id, \
    do_sample=True, torch_dtype=torch.bfloat16, max_new_tokens=128, device=0) #remove device=0 when using Dolly/other newer models, use device_map=auto above instead

  hf_pipe = HuggingFacePipeline(pipeline=pipe)
  # Set verbose=True to see the full prompt:
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt)

# COMMAND ----------

qa_chain = build_qa_chain()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that there are _many_ factors that affect how the language model answers a question. Most notable is the prompt template itself. This can be changed, and different prompts may work better or worse with certain models.
# MAGIC
# MAGIC The generation process itself also has many knobs to tune, and often it simply requires trial and error to find settings that work best for certain models and certain data sets. See this [excellent guide from Hugging Face](https://huggingface.co/blog/how-to-generate). 
# MAGIC
# MAGIC The settings that most affect performance are:
# MAGIC - `max_new_tokens`: longer responses take longer to generate. Reduce for shorter, faster responses
# MAGIC - `k`: (below): the number of chunks of text retrieved into the prompt. Longer prompts take longer to process
# MAGIC - `num_beams`: if using beam search, more beams increase run time more or less linearly

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the Chain for Simple Question Answering
# MAGIC
# MAGIC That's it! it's ready to go. Define a function to answer a question and pretty-print the answer, with sources.
# MAGIC
# MAGIC 🚨 Note:
# MAGIC Here we are using dolly without any fine tuning on a specialized dataset. As a result, similar to any other question/answering model, dolly's results are not reliable and can be factually incorrect.

# COMMAND ----------

def answer_question(question):
  similar_docs = db.similarity_search(question, k=2)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
  result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text']}</blockquote></p>"
  result_html += "<p><hr/></p>"
  for d in result["input_documents"]:
    source_id = d.metadata["source"]
    result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_id}\">{source_id}</a>)</blockquote></p>"
  displayHTML(result_html)

# COMMAND ----------

# MAGIC %md 
# MAGIC Try asking a question about cystic fibrosis!

# COMMAND ----------

answer_question("What are the primary drugs for treating cystic fibrosis?")

# COMMAND ----------

answer_question("What are the cystic fibrosis drugs that modulate the CFTR protein?")
