![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

# Biomedical Question Answering over Custom Datasets with 🦜️🔗 LangChain and Open Source LLMs on Hugging Face 🤗 

Large Language Models produce some amazing results, chatting and answering questions with seeming intelligence. But how can you get LLMs to answer questions about _your_ specific datasets? Imagine answering questions based on your company's knowledge base, docs or Slack chats. The good news is that this is easy with open-source tooling and LLMs. This example shows how to apply [LangChain](https://python.langchain.com/en/latest/index.html), Hugging Face `transformers`, and open source LLMs such as [MPT-7b-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) from MosaicML or [Falcon-7b-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) from the Technology Innovation Institute. This example can make use of any text-generation LLM or even OpenAI with minor changes. In this case, the data set is a set of freely available published papers in PDF format about cystic fibrosis from PubMed, but could be any corpus of text.

___
<james.mccall@databricks.com>

___

<img style="margin: auto; display: block" width="1200px" src="https://raw.githubusercontent.com/databricks-industry-solutions/hls-llm-doc-qa/basic-qa-LLM-HLS/images/solution-overview.jpeg?token=GHSAT0AAAAAACBNXSB43DCNDGVDBWWEWHZCZEBLWBA">

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| transformers                                 | Transformers ML     | Apache 2.0        | https://github.com/huggingface/transformers                      |
| sentence-transformers      | Embeddings with BERT    | Apache 2.0      | https://github.com/UKPLab/sentence-transformers  |
| langchain                                 | LLM Applications      | MIT        | https://github.com/hwchase17/langchain                      |
| chromadb                                 | Embedding Database      | Apache 2.0        | https://github.com/chroma-core/chroma                      |
| pypdf                                 | Reading PDF files      | Custom Open Source        | https://github.com/py-pdf/pypdf                      |
| pycryptodome                                 | Cryptographic library for Python      | BSD 2-Clause        | https://github.com/Legrandin/pycryptodome                      |
| accelerate                                 | Train and use PyTorch models with multi-GPU, TPU, mixed-precision      | Apache 2.0        | https://github.com/huggingface/accelerate                      |
| unstructured                                 | Build custom preprocessing pipelines for ML      | Apache 2.0        | https://github.com/yaml/pyyaml                      |
| sacremoses                                 | Tokenizer, truecaser and normalizer      | MIT        | https://github.com/hplt-project/sacremoses                      |
| ninja                                 | Small build system with a focus on speed      | Apache 2.0        | https://github.com/ninja-build/ninja                      |
| pytorch-lightning | Lightweight PyTorch wrapper | Apache 2.0 | https://github.com/Lightning-AI/lightning |
| xformers | Transformers building blocks | BSD 3-Clause | https://github.com/facebookresearch/xformers |
| triton | Triton language and compiler | MIT | https://github.com/openai/triton/ |


## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
