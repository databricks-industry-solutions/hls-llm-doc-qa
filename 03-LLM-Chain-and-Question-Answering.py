# Databricks notebook source
# MAGIC %md This notebook is available at https://github.com/databricks-industry-solutions/hls-llm-doc-qa

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Chain Creation
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
# MAGIC Start with required libraries for data preparation. There is one library in particular, FlashAttention, that takes ~5 minutes to install, so this will take a little bit.

# COMMAND ----------

# MAGIC %run ./util/install-llm-libraries

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a dropdown widget for model selection, as well as defining the file paths where our PDFs are stored, where we want to cache the HuggingFace model downloads, and where we want to persist our vectorstore.

# COMMAND ----------

# which LLM do you want to use? You can grab LLM names from Hugging Face and replace/add them here if you want
dbutils.widgets.dropdown('model_name','mosaicml/mpt-7b-instruct',['databricks/dolly-v2-7b','tiiuae/falcon-7b-instruct','mosaicml/mpt-7b-instruct'])

# which embeddings model from Hugging Face 🤗  you would like to use; for biomedical applications we have been using this model recently
# also worth trying this model for embeddings for comparison: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
dbutils.widgets.text("Embeddings_Model", "pritamdeka/S-PubMedBert-MS-MARCO")

# where was the vectorstore persisted in the previous notebook?
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/langchain_hls/db")

# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

#get widget values
db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")
embeddings_model = dbutils.widgets.get("Embeddings_Model")

# COMMAND ----------

import os
# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Load the Chroma DB and define our retriever. We define `k` here, which is how many chunks of text we want to retrieve from the vectorstore to feed into the LLM
# MAGIC - Instantiate an LLM, like Dolly here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

# COMMAND ----------

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

db_persist_path = db_persist_path
hf_embed = HuggingFaceEmbeddings(model_name=embeddings_model)
db = Chroma(collection_name="hls_docs", embedding_function=hf_embed, persist_directory=db_persist_path)

#k here is a particularly important parameter; this is how many chunks of text we want to retrieve from the vectorstore
retriever = db.as_retriever(search_kwargs={"k": 4})

# COMMAND ----------

 # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        # support for flast-attn and openai-triton is coming soon
        # config.attn_config['attn_impl'] = 'triton'
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.model.to(device='cuda')
        
        self.model.eval()

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

  config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
  
  if model_name == "mosaicml/mpt-7b-instruct":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") #for use with mpt-7b
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda:0' # For fast initialization directly on GPU!
  else:
    tokenizer = AutoTokenizer.from_pretrained(model_name) #for use with other models

  model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)

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

#this can take quite a while the first time you run this, as it must download the model from Hugging Face (can be many GB). These will be cached afterwards and get faster
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

answer_question("What are the primary drugs for treating cystic fibrosis (CF)?")

# COMMAND ----------

answer_question("What are the cystic fibrosis drugs that target the CFTR protein?")
