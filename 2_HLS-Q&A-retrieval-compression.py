# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/hls-llm-doc-qa

# COMMAND ----------

# MAGIC %md
# MAGIC # Question Answering over Custom Datasets with 🦜️🔗 LangChain and MPT-7b-Instruct from MosaicML on Hugging Face 🤗 
# MAGIC
# MAGIC Large Language Models produce some amazing results, chatting and answering questions with seeming intelligence. But how can you get LLMs to answer questions about _your_ specific datasets? Imagine answering questions based on your company's knowledge base, docs or Slack chats. The good news is that this is easy with open-source tooling and LLMs. This example shows how to apply [LangChain](https://python.langchain.com/en/latest/index.html), Hugging Face `transformers`, and an open source LLM from MosaicML called [MPT-7b-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct). This example can make use of any text-generation LLM or even OpenAI with minor changes. In this case, the data set is a set of freely available published papers in PDF format about cystic fibrosis from PubMed, but could be any corpus of text.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Setup
# MAGIC
# MAGIC - Run this on a cluster with Databricks Runtime 13.0 ML GPU. It should work on 12.2 ML GPU as well.
# MAGIC - To run this notebook's examples _without_ distributed Spark inference, all that is needed is a single-node 'cluster', with a single A10 GPU (ex: `g5.4xlarge` on AWS). A100 instances work as well.
# MAGIC
# MAGIC We have provided code to create a sample cluster for this notebook. See the `./RUNME` notebook for instructions on how to use this automation.
# MAGIC
# MAGIC Start with required Python libraries:

# COMMAND ----------

# MAGIC %run ./util/install-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

# which LLM do you want to use? You can grab LLM names from Hugging Face and replace/add them here if you want
dbutils.widgets.dropdown('model_name','mosaicml/mpt-7b-instruct',['databricks/dolly-v2-7b','databricks/dolly-v2-3b','mosaicml/mpt-7b-instruct'])

# where you want the PDFs to be saved in your environment
dbutils.widgets.text("PDF_Path", "/dbfs/tmp/langchain_hls/pdfs")

# where you want the vectorstore to be persisted across sessions, so that you don't have to regenerate
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/langchain_hls/db")

# publicly accessible bucket with PDFs for this demo
dbutils.widgets.text("Source_Documents", "s3a://db-gtm-industry-solutions/data/hls/llm_qa/")

#
dbutils.widgets.dropdown('performance_mode','fast',['fast','quality'])

# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

#get widget values
pdf_path = dbutils.widgets.get("PDF_Path")
source_pdfs = dbutils.widgets.get("Source_Documents")
db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")

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

import os
import shutil

# in case you rerun this notebook, this deletes the directory and recreates it to prevent file duplication
if os.path.exists(pdf_path):
  shutil.rmtree(pdf_path)
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
# MAGIC - When we create embeddings for these documents, an NLP creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - More info on embeddings: [Hugging Face: Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)
# MAGIC - More info on Text Splitting with LangChain: [Text Splitters](https://python.langchain.com/en/latest/modules/indexes/text_splitters.html)
# MAGIC - Great article on [text chunking strategies for LLM apps](https://www.pinecone.io/learn/chunking-strategies/) from Pinecone

# COMMAND ----------

# For PDFs we need to split them for embedding:
from langchain.text_splitter import NLTKTextSplitter
text_splitter = NLTKTextSplitter(chunk_size=2000)
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

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(collection_name="hls_docs", embedding_function=hf_embed, persist_directory=db_persist_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Tuning
# MAGIC Here we are enabling some performance tuning of the model; there is much, much more that can be modified to enable faster inference or higher quality responses, but these are some of the higher impact options.
# MAGIC
# MAGIC - If you select `quality`, the chain can retrieve more text chunks (more context), produce more text, and produce higher quality text; expect it to take 4-5 minutes per question with those settings.
# MAGIC - If you select `fast`, the answer will be shorter/lower quality and will have less context, but should complete in less than a minute.
# MAGIC
# MAGIC The generation process itself has many knobs to tune, and often it simply requires trial and error to find settings that work best for certain models and certain data sets. See this [excellent guide from Hugging Face](https://huggingface.co/blog/how-to-generate). 
# MAGIC
# MAGIC The settings that most affect performance are:
# MAGIC - `max_new_tokens`: longer responses take longer to generate. Reduce for shorter, faster responses
# MAGIC - `k`: (below): the number of chunks of text retrieved into the prompt. Longer prompts take longer to process
# MAGIC - `num_beams`: if using beam search, more beams increase run time more or less linearly

# COMMAND ----------

perf_mode = dbutils.widgets.get("performance_mode")

if perf_mode == "fast":
  k_chunks = 2
  beams = 1
  max_new_tokens = 128
  input_max_seq_len = 2048 #only for use with MPT-7b; if you change LLMs, comment this out and remove this config item from build_qa_chain below
else:
  k_chunks = 4
  beams = 4 #this can be set higher if you are willing to have higher inference time
  max_new_tokens = 256
  input_max_seq_len = 4096 #only for use with MPT-7b; if you change LLMs, comment this out and remove this config item from build_qa_chain below

#print(k_chunks, beams, max_new_tokens, input_max_seq_len)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating our Vectorstore and defining document retrieval
# MAGIC Here we are creating a retriever to retrieve the most relevant documents using MMR (maximum marginal relevance), and a filter that will only keep documents above a certain similarity threshold. This `compression_retriever` is used later on when we ask questions.

# COMMAND ----------

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

retriever = db.as_retriever(search_kwargs={"k": k_chunks}, search_type="mmr")

redundant_filter = EmbeddingsRedundantFilter(embeddings=hf_embed)
relevant_filter = EmbeddingsFilter(embeddings=hf_embed, similarity_threshold=0.60)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Instantiate an LLM, like Dolly here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

# COMMAND ----------

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

def build_qa_chain():
  torch.cuda.empty_cache() # Not sure this is helping in all cases, but can free up a little GPU mem
  model_name=dbutils.widgets.get('model_name') #selected from the dropdown widget at the top of the notebook
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") #for use with mpt-7b

  config = transformers.AutoConfig.from_pretrained('mosaicml/mpt-7b-instruct', trust_remote_code=True) #for use with mpt-7b
  config.update({"max_seq_len": input_max_seq_len})
  model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True) #for use with mpt-7b

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
    do_sample=True, torch_dtype=torch.bfloat16, max_new_tokens=max_new_tokens, num_beams=beams, device=0) #remove device=0 when using Dolly/other newer models, use device_map=auto above instead

  hf_pipe = HuggingFacePipeline(pipeline=pipe)
  # Set verbose=True to see the full prompt:
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt)

# COMMAND ----------

qa_chain = build_qa_chain()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that there are _many_ factors that affect how the language model answers a question. Most notable is the prompt template itself. This can be changed, and different prompts may work better or worse with certain models.

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
  similar_docs = compression_retriever.get_relevant_documents(question)
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

# COMMAND ----------


