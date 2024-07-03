# CyberQA
A benchmark for evluating cyber security knowledge evaluation on LLM.

There are 3 main datasets:
* manually curated QA from a mix of sources
* auto created QA from a mix of sources
* human benchmark: in progress

Each QA is tagged with a difficulty level ranging from level 1 which is factual knowledge  to level 5 which involves some form of reasoning.
The goal is to compare the human baseline to various LLM models pre-trained or with a RAG backend.

# Sources material

https://github.com/CSIRT-MU/edu-resources

# Setup

Install the python dependencies.

Login into deepeval: deepeval login

# Architecture

Based on deepeval
# Submodules

git submodule add git@github.com:microsoft/Security-101.git Security-101

# Dependencies

Deepeval
Langsmith
