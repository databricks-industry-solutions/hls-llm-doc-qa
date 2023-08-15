# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/hls-llm-doc-qa

# COMMAND ----------

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

# which LLM do you want to use? Grab the name of the model you deployed in step 04 from Databricks Model Serving
dbutils.widgets.text('model_name_from_model_serving',"llama-2-7b-chat")

# which embeddings model from Hugging Face 🤗  you would like to use; for biomedical applications we have been using this model recently
# also worth trying this model for embeddings for comparison: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
dbutils.widgets.text("Embeddings_Model", "pritamdeka/S-PubMedBert-MS-MARCO")

# where was the vectorstore persisted in the previous notebook?
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/langchain_hls/db")

# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

#get widget values
model_name=dbutils.widgets.get('model_name_from_model_serving')
db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")
embeddings_model = dbutils.widgets.get("Embeddings_Model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Load the Chroma DB and define our retriever. We define `k` here, which is how many chunks of text we want to retrieve from the vectorstore to feed into the LLM
# MAGIC - Instantiate an LLM, loading from Databricks Model serving here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

# COMMAND ----------

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

db_persist_path = db_persist_path
hf_embed = HuggingFaceEmbeddings(model_name=embeddings_model)
db = Chroma(collection_name="hls_docs", embedding_function=hf_embed, persist_directory=db_persist_path)

#k here is a particularly important parameter; this is how many chunks of text we want to retrieve from the vectorstore
retriever = db.as_retriever(search_kwargs={"k": 3})

# COMMAND ----------

# If running a Databricks notebook attached to an interactive cluster in "single user"
# or "no isolation shared" mode, you only need to specify the endpoint name to create
# a `Databricks` instance to query a serving endpoint in the same workspace.

# Otherwise, you can manually specify the Databricks workspace hostname and personal access token
# or set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, respectively.

from langchain.llms import Databricks

#llm = Databricks(endpoint_name=model_name)
llm = Databricks(endpoint_name=model_name, model_kwargs={"temperature": 0.1,"max_tokens": 512})

#if you want answers to generate faster, set the number of tokens above to a smaller number
prompt = "What is cystic fibrosis?"
#sample question, if you want to try it out
displayHTML(llm(prompt))

# COMMAND ----------

# MAGIC %md
# MAGIC If you were not able to deploy to GPU serving in the previous step but you were able to register the model with MLflow, you can load the model from the registry using the below code. This will require you to run this notebook on a GPU cluster.
# MAGIC

# COMMAND ----------


'''
import mlflow
import pandas as pd
loaded_model = mlflow.pyfunc.load_model(f"models:/llama2-7b-MedText-QLoRA/latest")

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
print(loaded_model.predict(input_example))

'''

# COMMAND ----------

from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def build_qa_chain():
  
  template = """You are a life sciences researcher with deep expertise in cystic fibrosis and related comorbidities. Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  Use only information in the following paragraphs to answer the question. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
  
  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  # Set verbose=True to see the full prompt:
  return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

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

def answer_question(question):
  similar_docs = retriever.get_relevant_documents(question)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
  #result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text']}</blockquote></p>" #depending on which prompt template you use, different response parsing might be needed
  result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text'].split('### Response')[1].strip()}</blockquote></p>"
  result_html += "<p><hr/></p>"
  for d in result["input_documents"]:
    source_id = d.metadata["source"]
    result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_id}\">{source_id}</a>)</blockquote></p>"
  displayHTML(result_html)

# COMMAND ----------

question = "What are the primary drugs for treating cystic fibrosis (CF)?"
similar_docs = retriever.get_relevant_documents(question)
result = qa_chain({"input_documents": similar_docs, "question": question})
result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text'].split('### Response')[1].strip()}</blockquote></p>"
result_html += "<p><hr/></p>"
for d in result["input_documents"]:
    source_id = d.metadata["source"]
    result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_id}\">{source_id}</a>)</blockquote></p>"
displayHTML(result_html)

# COMMAND ----------

# MAGIC %md 
# MAGIC Try asking a question about cystic fibrosis!

# COMMAND ----------

answer_question("What are the primary drugs for treating cystic fibrosis (CF)?")

# COMMAND ----------

answer_question("What are the cystic fibrosis drugs that target the CFTR protein?")

# COMMAND ----------


