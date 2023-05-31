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

# MAGIC %run ./util/install-prep-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

# which LLM do you want to use? You can grab LLM names from Hugging Face and replace/add them here if you want
dbutils.widgets.dropdown('model_name','mosaicml/mpt-7b-instruct',['databricks/dolly-v2-7b','databricks/dolly-v2-3b','mosaicml/mpt-7b-instruct'])

# where you want the vectorstore to be persisted across sessions, so that you don't have to regenerate
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/langchain_hls/db")

#Select a performance mode for the chain - fast/lower quality or higher quality/slower
dbutils.widgets.dropdown('performance_mode','fast',['fast','quality'])

# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

import os
# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# COMMAND ----------

# MAGIC %md
# MAGIC Loading our Vectorstore knowledge base from the previous notebook:

# COMMAND ----------

db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")

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
model_name = dbutils.widgets.get("model_name")

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
# MAGIC Running the LLM of choice on the driver node as an API on port 7777. The time for this to run depends on the model, but can take up to 15 minutes; 7-10 minutes to install special Nvidia packages (in the case of MosaicML's MPT-7b model), and 5-10 minutes for the LLM to download from Hugging Face and for the API to get set up.

# COMMAND ----------

#%run ./Driver_Proxy_setup $model_name="mosaicml/mpt-7b-instruct" $num_beams=1 $max_new_tokens=128

# COMMAND ----------

#dbutils.notebook.run("Driver_Proxy_setup", {"model_name": "mosaicml/mpt-7b-instruct", "num_beams": "1", "max_new_tokens": "128"})

# COMMAND ----------

from langchain.llms import Databricks

# COMMAND ----------

# Use `transform_input_fn` and `transform_output_fn` if the app
# expects a different input schema and does not return a JSON string,
# respectively, or you want to apply a prompt template on top.

def transform_input(**request):
    full_prompt = f"""{request["prompt"]}
    Be Concise.
    """
    request["prompt"] = full_prompt
    return request

def transform_output(response):
    return response.upper()

llm = Databricks(
  cluster_driver_port="7777",
  transform_input_fn=transform_input,
  transform_output_fn=transform_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating our Vectorstore and defining document retrieval
# MAGIC Here we are creating a retriever to retrieve the most relevant documents using MMR (maximum marginal relevance), and a filter that will only keep documents above a certain similarity threshold. This `compression_retriever` is used later on when we ask questions.

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# COMMAND ----------

#testing compression retriever

#retriever = db.as_retriever(search_kwargs={"k": 4}, search_type="mmr")
retriever = db.as_retriever(search_kwargs={"k": 4})
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# COMMAND ----------

#question="How is cycstic fibrosis (CF) diagnosed, and what are the primary indicators?"
#docs = retriever.get_relevant_documents(question)
#display(docs)

# COMMAND ----------

#similar_docs = compression_retriever.get_relevant_documents(question)
#print(similar_docs)

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
  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  Use only information in the following paragraphs to answer the question. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
  
  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# COMMAND ----------

qa_chain = build_qa_chain()

# COMMAND ----------

from langchain.chains.question_answering import load_qa_chain
qa_chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)

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


