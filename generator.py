
from dotenv import load_dotenv
import uuid
import csv
import argparse
import typing
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

load_dotenv()  # take environment variables from .env.

def deepeval_generator(file_path:str,dataset_name:str)->str:
    """

    :param file_path:
    :param db_name:
    :return: the url to the database
    """
    from deepeval.dataset import EvaluationDataset
    dataset = EvaluationDataset()
    dataset.add_test_cases_from_csv_file(
        # file_path is the absolute path to you .csv file
        file_path=file_path,
        input_col_name="Question",
        actual_output_col_name="Answer"
    )

    result = dataset.push(alias=dataset_name,overwrite=True)

def langsmith_generator(file_path:str,dataset_name:str)->str:
    """

    :param file_path:
    :param db_name:
    :return: the url to the database
    """
    from langsmith import Client
    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(dataset_name=dataset_name)

    with open(file_path, 'r') as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            client.create_example(
                inputs={"question": row["Question"]},
                outputs={"answer": row["Answer"]},
                metadata={"source": row['Source'],"level":row['Level']},
                dataset_id=dataset.id
            )

    return dataset.url

parser = argparse.ArgumentParser(description="QA generator from curated dataset")

parser.add_argument('-action', help='The action to take (e.g. generate)',default="generate")
parser.add_argument('-backend', help='The action to take (e.g. generate)',default="langsmith")
parser.add_argument('-source', help='The action to take (e.g. generate)',default="./Curated/various_qa.csv")
parser.add_argument('-name', help='The action to take (e.g. generate)',default="test QA")

args = parser.parse_args()

if args.action == "generate":
    if args.backend == "langsmith":
        dataset_url = langsmith_generator(args.source,args.name)
        logging.info("Dataset is ready %s" % dataset_url)
    elif args.backend == "deepeval":
        dataset_url = deepeval_generator(args.source,args.name)
        logging.info("Dataset is ready %s" % dataset_url)
else:
    logging.warning("Not supported")