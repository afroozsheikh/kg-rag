from abc import ABC
import os
import re
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Prompt,
    PromptTemplate,
)

from llama_index.core import StorageContext, load_index_from_storage


class LlamaIndexPipeline(ABC):
    def __init__(self, knowledge_dict, reports_path) -> None:
        self.knowledge_dict = knowledge_dict
        self.reports_path = reports_path

    def count_subfolders(self, folder_path):
        subfolder_count = 0
        subfolder_paths = []

        for root, dirs, files in os.walk(folder_path):
            if root != folder_path:
                subfolder_count += 1
        basepath = "/Users/afrouz.sheikholeslami/Macquarie/Thesis/Implementations/HealthLLM/dataset_folder/health_report_"
        for i in range(subfolder_count):
            path_rr = basepath + str({i})
            subfolder_paths.append(path_rr)

        return subfolder_count, subfolder_paths

    # def generate_prompt(self, patient_report_path, query, domain_knowledge):
    def generate_prompt(self, patient_report_path, domain_knowledge):
        """A function for prompt generation

        Args:
            patient_report_path (str): path to the folder of the patient's medical report
            query (str): the question you wanna ask
            domain_knowledge (str): _description_

        Returns:
            str: _description_
        """

        try:
            with open(patient_report_path, "r", encoding="utf-8") as file:
                medical_report = file.read()
        except FileNotFoundError:
            return "Error: The medical report file was not found."

        prompt = (
            "Here is some additional professional health knowledge that can help you better analyze the report:\n"
            "----------------------------------------------------------------------\n"
            f"{domain_knowledge}\n"
            "----------------------------------------------------------------------\n"
            "Give the answer in JSON format with only one floating point number between 0 and 1 that is “score”. "
            "The rule of the answer: 0-0.2 is mild or none, 0.3-0.6 is moderate, and above 0.7 is severe.\n"
            "This is a patient’s medical record. Context information:\n"
            "----------------------------------------------------------------------\n"
            f"{medical_report}\n"
            "----------------------------------------------------------------------\n"
            "Given the context information, you are a helpful health consultant, so answer the question:\n"
            # f"{query}\n"
        )

        return prompt

    def extract_scores(self, string):
        numbers = re.findall(r"\d+\.\d+|\d+", string)
        if numbers:
            for i in numbers:
                return float(i)
        else:
            return 0.0

    def read_reports(self, path):
        reports = []
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                filepath = os.path.join(path, filename)

                # Read the .docx file
                with open(filepath, "r") as f:
                    txt = f.read()
                    reports.extend(txt.split("\n"))

        return reports

    def run(self):
        num_patients = self.count_subfolders(self.reports_path)
        patients_scores = {}
        for i in range(num_patients):
            score_ls = []
            medical_report = SimpleDirectoryReader(
                f"{self.reports_path}/health_report_{i}"
            ).load_data()
            index = VectorStoreIndex.from_documents(medical_report)
            index.set_index_id("index_health")
            for query, domain_knowledge in self.knowledge_dict.values():
                prompt = self.generate_prompt(
                    patient_report_path=medical_report,
                    domain_knowledge=domain_knowledge,
                )
                query_engine = index.as_query_engine()
                response = str(query_engine.query(query))
                score_ls.append(self.extract_scores(response))

            patients_scores[i] = score_ls
        return patients_scores
