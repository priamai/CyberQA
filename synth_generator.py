"""
Generate the QA golden rules
"""
import os

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

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
    generate_microsoft()
else:
    print("You asked for something other than installation")

