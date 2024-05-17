import openai
import argparse
from repository.retrieval import RetrievalPipeline
from interface.llamaindex import LlamaIndexPipeline
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Resnet_v2 model"
    )
    parser.add_argument(
        "--disease_path",
        type=str,
        default="/Users/afrouz.sheikholeslami/Macquarie/Thesis/Implementations/HealthLLM/Input_feature.txt",
        required=False,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--reports_path",
        type=str,
        default="/Users/afrouz.sheikholeslami/Macquarie/Thesis/Implementations/HealthLLM/dataset_folder",
        required=False,
    )
    parser.add_argument(
        "--knowledge_path",
        type=str,
        default="/Users/afrouz.sheikholeslami/Macquarie/Thesis/Implementations/HealthLLM/exsit_knowledge/my_dict.pkl",
        required=False,
    )

    args = parser.parse_args()
    return args


def main(args):
    os.environ["OPENAI_API_KEY"] = ""
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    retrieval = RetrievalPipeline(
        diseases_path=args.disease_path, knowledge_path=args.knowledge_path, k=args.k
    )
    knowledge_dict = retrieval.run()

    llama_index = LlamaIndexPipeline(
        knowledge_dict=knowledge_dict, reports_path=args.reports_path
    )
    patients_scores = llama_index.run()


if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
