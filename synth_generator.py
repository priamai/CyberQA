"""
Generate the QA golden rules
"""
import os
import tiktoken
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
import glob
from pathlib import Path
from dotenv import load_dotenv
import re
import csv
load_dotenv()  # take environment variables from .env.

def parse_csv_markdown(csv_string: str) -> dict:
    # Try to find JSON string within first and last triple backticks
    match = re.search(r"""```       # match first occuring triple backticks
                          (?:csv)? # zero or one match of string json in non-capturing group
                          (.*)```   # greedy match to last triple backticks""", csv_string, flags=re.DOTALL|re.VERBOSE)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        csv_str = csv_string
    else:
        # If match found, use the content within the backticks
        csv_str = match.group(1)

    # Strip whitespace and newlines from the start and end
    csv_str = csv_str.strip()

    return csv_str


def langchain_markdown(file_path: str,model:str="gpt-4-turbo") -> str:
    """

    :param file_path:
    :param db_name:
    :return: the url to the database
    """
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain_core.documents import Document
    from langchain.globals import set_llm_cache
    from langchain_openai import ChatOpenAI
    import openai
    from langchain_core.prompts import ChatPromptTemplate
    import io
    # We can do the same thing with a SQLite cache
    from langchain.cache import SQLiteCache

    #set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    document_paths = []
    # list all the markdown files
    for mdfile in glob.glob(f"{file_path}/*.md"):
        path = Path(mdfile)
        if path.name[0].isdigit():
            document_paths.append(mdfile)

    print("Ready to generate QA from %d files" % len(document_paths))
    all_qas = []
    for markdown_path in document_paths:
        loader = UnstructuredMarkdownLoader(markdown_path)

        data = loader.load()
        assert len(data) == 1
        assert isinstance(data[0], Document)
        text = data[0].page_content
        #check the document size to make sure we don't go over the limit
        print("File: %s" % markdown_path)
        text_size = len(text)
        tokens = num_tokens_from_string(text, model)
        print("Total: Text %d Tokens %d " % (text_size,tokens))
        if tokens > 128000/2:
            print("Too many tokens")
            continue

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are lecturer in the computer science department for cyber security and"
                    "need to write a questionnaire based on the following textbook for an exam."
                    "For each text provided below can you extract a set a pair of questions and answers?"
                    "Please format it as a CSV file where the header: Question,Answer,Level,Source"
                    "Where Source is {source} and Level is {level}, Question and Answer are the ones generated."
                    "Always quote strings."
                    "If there are no questions and answers pairs then produce just the header of the CSV",
                ),
                ("human", "{content}"),
            ]
        )

        chain = prompt | llm


        result = chain.invoke(
            {
                "content": text,
                "level": "1",
                "source": "Microsoft Security 101"
            }
        )

        parsed = parse_csv_markdown(result.content)
        sio = io.StringIO(parsed)
        reader = csv.DictReader(sio,delimiter=',', quotechar='"')

        for row in reader:
            all_qas.append(row)

    with open("./Curated/auto_microsoft_101.csv","w") as csvfile:
        fieldnames = ['Question','Answer','Level','Source']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=',', quotechar='"')

        writer.writeheader()
        for row in all_qas:
            writer.writerow(row)

def generate_microsoft(model="gpt-4o",max_questions=10):
    deepeval.login_with_confident_api_key(os.environ["CONFIDENT_API_KEY"])
    dataset = EvaluationDataset()

    document_paths = []
    # list all the markdown files
    for mdfile in glob.glob("Security-101/*.md"):
        path = Path(mdfile)
        if path.name[0].isdigit():
            document_paths.append(mdfile)
    print("Ready to generate QA from %d files" % len(document_paths))
    # Use a model
    synthesizer = Synthesizer(model=model)

    dataset.generate_goldens_from_docs(
        synthesizer=synthesizer,
        document_paths=document_paths,
        max_goldens_per_document=max_questions
    )
    print("Pushing to cloud...")
    dataset.push(alias="security101")

import argparse

parser = argparse.ArgumentParser(description="QA generator")

parser.add_argument('-action', help='The action to take (e.g. generate)')

args = parser.parse_args()

if args.action == "generate":
    langchain_markdown("./Security-101")
else:
    print("You asked for something other than installation")

