from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="./Curated/nist_sp_800-61r2.csv",
    input_col_name="question",
    actual_output_col_name="response"
)

dataset.push(alias="example_ninst",overwrite=True)

