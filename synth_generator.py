"""
Generate the QA golden rules
"""
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
import glob
from pathlib import Path

dataset = EvaluationDataset()

document_paths = []
# list all the markdown files
for mdfile in glob.glob("Security-101/*.md"):
    path = Path(mdfile)
    if path.name[0].isdigit():
        document_paths.append(mdfile)


# Use gpt-3.5-turbo instead
synthesizer = Synthesizer(model="gpt-3.5-turbo")


dataset.generate_goldens_from_docs(
    synthesizer=synthesizer,
    document_paths=['example.pdf'],
    max_goldens_per_document=2
)
