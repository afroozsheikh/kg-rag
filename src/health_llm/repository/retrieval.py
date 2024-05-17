from abc import ABC
import openai
import torch
from sentence_transformers import SentenceTransformer
from repository.hopfield import HopfieldRetrievalModel
import os
import pickle


class RetrievalPipeline(ABC):
    def __init__(self, diseases_path, knowledge_path, k) -> None:
        self.diseases_path = diseases_path
        self.knowledge_path = knowledge_path

        self.k = k

    def read_diseases(self, disease_path):
        with open(disease_path, "r") as file:
            disease_ls = [line.strip() for line in file]

        return disease_ls

    def generate_questions(self, disease_ls):
        questions = []
        for i in disease_ls:
            questions.append(
                f"Does the person described in the case have {i} symptoms? Do you think it is serious?"
            )
        return questions

    def generate_disease_symptoms(self, disease_ls):
        symptoms = []
        for i in disease_ls:
            openai.api_key = "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"
            messages = []
            system_message = (
                "please list the names of the symptoms in order like these examples\n"
                + "Cold ------ Runny "
                "nose, stuffy nose, "
                "sneezing, "
                "sore throat, "
                "cough, hoarseness, "
                "headache, "
                "sore eyes, "
                "fatigue, "
                "minor body aches, "
                "fever or low-grade "
                "fever, ear pain, "
                "chest tightness, "
                "or difficulty "
                "breathing."
            )
            messages.append({"role": "system", "content": system_message})
            message = f"Please list the symptoms of {i}? in the following format"
            messages.append({"role": "user", "content": message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            symptoms.append(response["choices"][0]["message"]["content"])
        return symptoms

    def read_domain_knowledge(self, knowledge_path):
        """_summary_

        Args:
            path (_type_): path to pickle file

        Returns:
            _type_: _description_
        """
        with open(knowledge_path, "rb") as file:
            loaded_data = pickle.load(file)
        knowledge_paragraphs = []
        for i in loaded_data:
            knowledge_paragraphs.append(loaded_data[i])

        return knowledge_paragraphs

    def retrieve(self, query, paragraphs, k):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer("all-mpnet-base-v2")

        p_embeddings = model.encode(paragraphs)

        query_embeddings = model.encode(query)

        retrievaler = HopfieldRetrievalModel().to(device)
        result = retrievaler(
            torch.tensor(p_embeddings).to(device) * 100,
            torch.tensor(query_embeddings).to(device) * 100,
        )
        input_ids = torch.topk(result, k, dim=1).indices
        indices = input_ids[0]

        knowledge = []
        for indice in indices:
            knowledge.append(paragraphs[indice])
        knowledge = [x for x in knowledge if x != ""]
        return knowledge

    def run(self):
        disease_ls = self.read_diseases(self.diseases_path)
        questions = self.generate_questions(disease_ls=disease_ls)
        knowledge_paragraphs = self.read_domain_knowledge(self.knowledge_path)
        knowledge_dict = {
            question: self.retrieve(
                query=question, paragraphs=knowledge_paragraphs, k=self.k
            )
            for question in questions
        }
        return knowledge_dict
