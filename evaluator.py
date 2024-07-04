
from dotenv import load_dotenv
import uuid
import csv
import argparse
import typing
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

load_dotenv()  # take environment variables from .env.

def langsmith_evaluator(dataset_name:str,evaluator_llm:str="gpt-4-turbo",ai_model:str="gpt-3.5-turbo-16k",backend:str="OpenAI",temperature=0):
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from operator import itemgetter
    from langchain.smith import RunEvalConfig
    from langsmith import Client
    client = Client()

    if client.has_dataset(dataset_name=dataset_name):
        logging.info("Loaded dataset")
        dataset = client.read_dataset(dataset_name=dataset_name)
        logging.info("Created %s " % dataset.created_at)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful documentation Q&A assistant, trained to answer"
                    " questions from the cyber security domain",
                ),
                ("human", "{question}"),
            ]
        )

        # the model to test
        model = ChatOpenAI(model=ai_model, temperature=0)
        response_generator = prompt | model | StrOutputParser()

        # the evaluator LLM
        eval_config = RunEvalConfig(
            evaluators=["qa"],
            eval_llm= ChatOpenAI(model=evaluator_llm, temperature=0),

        )

        chain = ({"question":itemgetter("question")})| response_generator

        result = client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
            project_metadata={"temperature":temperature,"backend":backend}
        )

        project_name = result["project_name"]
        score = 0
        for rid,info in result['results'].items():
            feedback = info['feedback']
            for eval in feedback:
                score += eval.score
                comment = eval.comment
                key = eval.key

        return (project_name,score)



def deepeval_evaluator(dataset_name:str):
    raise NotImplemented("Not ready!")

parser = argparse.ArgumentParser(description="QA evaluator")

parser.add_argument('-action', help='The action to take (e.g. generate)',default="evaluate")
parser.add_argument('-backend', help='The action to take (e.g. generate)',default="langsmith")
parser.add_argument('-name', help='The name of the dataset',default="test QA")

args = parser.parse_args()

if args.action == "evaluate":
    if args.backend == "langsmith":
        (eval_id, score)= langsmith_evaluator(args.name)
        logging.info("Eval ID {0} Score {1}".format(eval_id,score))
    elif args.backend == "deepeval":
        (eval_id, score)= deepeval_evaluator(args.name)
        logging.info("Eval ID {0} Score {1}".format(eval_id,score))
else:
    logging.warning("Not supported")
