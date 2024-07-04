# CyberQA
A benchmark for evluating cyber security knowledge evaluation on LLM.

There are 3 main datasets:
* manually curated QA from a mix of sources
* auto created QA from a mix of sources
* human benchmark: in progress

Each QA is tagged with a difficulty level ranging from level 1 which is factual knowledge  to level 5 which involves some form of reasoning.
The goal is to compare the human baseline to various LLM models pre-trained or with a RAG backend.

# Human Benchmark

I am looking for volunteers to answers those questions, work in progress.

# How to run

The generator script creates the datasets based on source material.

``
python generator.py -action generate
``

The evaluator script performs the benchmark for a set of LLM models.


``
python evaluator.py -action evaluate
``

Make sure you set all the env variables.

# Sources material

https://github.com/CSIRT-MU/edu-resources

# Setup

Install the python dependencies via poetry or pip.

Login into deepeval: deepeval login

# Architecture

It relies on Langchain, Langsmith and DeepEval (choose one)

# Submodules

Security-101.git Security-101

# Dependencies

Deepeval
Langsmith
Langchain
