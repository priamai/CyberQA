# We have some hard-coded examples here.
from langsmith import Client
from dotenv import load_dotenv
import uuid
import csv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.smith import RunEvalConfig

from datetime import datetime

load_dotenv()  # take environment variables from .env.

client = Client()

dataset_name = f"Cyber QA Questions {str(uuid.uuid4())}"
dataset = client.create_dataset(dataset_name=dataset_name,description="Test GPT 3.5 Turbo in Cyber questions")
filename = "../Curated/various_qa.csv"

with open(filename, 'r') as csvfile:
    datareader = csv.DictReader(csvfile)
    for row in datareader:
        client.create_example(
            inputs={"question": row["Question"]}, outputs={"answer": row["Answer"]}, dataset_id=dataset.id
        )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful documentation Q&A assistant, trained to answer"
            " questions from the cyber security domain",
        ),
        ("human", "{question}"),
    ]
).partial(time=str(datetime.now()))

# the model to test
model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
response_generator = prompt | model | StrOutputParser()

# the evaluator LLM
eval_config = RunEvalConfig(
    evaluators=["qa"],
    eval_llm= ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
)

chain = ({"question":itemgetter("question")})| response_generator

result = client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=lambda: chain,
    evaluation=eval_config,
)

print(result)
