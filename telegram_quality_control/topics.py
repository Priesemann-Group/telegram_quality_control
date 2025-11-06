import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional
import os

try:
    import openai
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, OpenAI
    from cuml import UMAP
    from cuml.cluster import HDBSCAN
    from sentence_transformers import SentenceTransformer
    # from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    WITH_GPU_SUPPORT = True

except ImportError:
    logging.warning(
        "GPU-based dependencies are not installed -- you can use this module to "
        "load topics, but not to create new ones. To enable full functionality, "
        "install GPU dependencies with: `poetry install --with ai`"
    )
    WITH_GPU_SUPPORT = False

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from dotenv import dotenv_values


def get_cache_folder():
    config = {
        **dotenv_values("example.env"),
        **dotenv_values(".env"),
    }
    cache_folder = os.environ.get("SCRATCH_FOLDER", config.get("SCRATCH_FOLDER"))
    if cache_folder is None:
        raise ValueError("SCRATCH_FOLDER is not set in .env file")
    cache_folder = Path(cache_folder)

    return cache_folder

@dataclass
class Embeddings:
    
    folder_tag: str
    
    # Choose one of the pretrained models from
    # https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    # You can also find more embeddings here: https://huggingface.co/spaces/mteb/leaderboard
    embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    
    obj: Optional[np.ndarray] = None

    def create(self, docs):
        if not WITH_GPU_SUPPORT:
            raise RuntimeError(
                "GPU-based dependencies not available - embedding creation is disabled. "
                "To enable full functionality, install GPU dependencies with: "
                "poetry install --with ai"
            )

        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode(docs, show_progress_bar=True)
        self.obj = embeddings
        return embeddings
    
    @property
    def cache_folder(self):
        folder = get_cache_folder()
        return folder / "embeddings" / self.folder_tag

    @classmethod
    def load_from_file(cls, path):
        path = Path(path)
        embedding_path = path / "embeddings.npy"
        return np.load(embedding_path)

    def save(self):
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        path = self.cache_folder / "embeddings.npy"
        np.save(path, self.obj)

    # def iter_chats(self) -> Iterable[tuple[np.ndarray, pd.DataFrame]]:
    #     """Returns the embeddings for each chat

    #     Shapes might differ.
    #     """
    #     message_path = CleanMessages(self.params).cache_folder / "messages.parquet"
    #     messages_df = pd.read_parquet(
    #         message_path, columns=["date", "views", "forwards", "chat_id"]
    #     )
    #     messages_df["embedding_index"] = range(len(messages_df))
    #     all_embeds = self.load()
    #     chat_ids = messages_df["chat_id"].unique()
    #     for chat_id in chat_ids:
    #         messages = messages_df[messages_df["chat_id"] == chat_id]
    #         embeds = all_embeds[messages["embedding_index"].values]
    #         yield embeds, messages


@dataclass
class Topics:
    folder_tag: str
    
    # Dimensionality reduction
    num_neighbors: int = 15
    num_components: int = 5
    min_dist: float = 0.0
    dim_reduction_metric: str = "cosine"
    # Set to True if you have problems with out of memory errors
    low_memory: bool = False

    # Clustering
    min_cluster_size: int = 10  # 15
    min_samples: int = 10  # 15
    clustering_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True

    # Other topic model parameters
    calculate_probabilities: bool = True
    text_language: str = "english"
    
    obj: tuple = (None, None, None)

    def create(
        self,
        docs,
        embeddings,
        embedding_model,
        **kwargs,
    ):
        if not WITH_GPU_SUPPORT:
            raise RuntimeError(
                "GPU-based dependencies not available - topic modelling is disabled. "
                "To enable full functionality, install GPU dependencies with: "
                "poetry install --with ai"
            )
        
        topic_model = self.create_topic_model(embedding_model)

        topics, probs = topic_model.fit_transform(docs, embeddings)
        self.obj = topic_model, topics, probs
        return topic_model, topics, probs
    
    def create_topic_model(self, embedding_model):
        # 1. Embedding
        # Embed documents in a space
        embedding_model = SentenceTransformer(embedding_model)

        # 2. Dimensionality reduction
        # Reduce the dimensionality of the embedded space
        umap_model = UMAP(
            n_neighbors=self.num_neighbors,
            n_components=self.num_components,
            min_dist=self.min_dist,
            metric=self.dim_reduction_metric,
            # low_memory=self.low_memory,
        )

        # 3. Clustering
        # Cluster reduced embeddings
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.clustering_metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=self.prediction_data,
        )

        # 4. Tokenization
        # Create a representation for the cluster
        stop_words = np.loadtxt(
            f"./resources/stop_words/{self.text_language}.csv",
            delimiter="\t",
            dtype=str,
            encoding="utf-8",
        )
        vectorizer_model = CountVectorizer(stop_words=stop_words.tolist())

        # 5. Weight tokens
        # Using the default here

        # 5. Representat topics
        # Create a human-readable representation for the cluster
        # KeyBERTInspired representation
        BERTinspired_representation = KeyBERTInspired()

        # GPT representation
        # Check if the OPENAI_API_KEY is set in the .env file
        config = {
            **dotenv_values("example.env"),
            **dotenv_values(".env"),
        }
        try:
            api_key = config["OPENAI_API_KEY"]
        except KeyError:
            api_key = None

        if api_key is None or len(api_key) == 0:
            logging.warning(
                "OPENAI_API_KEY not set in .env file -- using simple topic labels instead of GPT labels."
            )
            use_gpt = False
        else:
            use_gpt = True

        if use_gpt:
            gpt_representation = self.get_gpt_representation_model(api_key)

            representation_model = {
                "Main": gpt_representation,
                "BERTinspired": BERTinspired_representation,
            }
        else:
            representation_model = BERTinspired_representation

        # Combine the models
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            calculate_probabilities=self.calculate_probabilities,
            language=self.text_language,
        )

        return topic_model

    @classmethod
    def load_from_file(cls, path):
        path = Path(path)
        model_path = path / "model"
        topics_path = path / "topics.npy"
        probs_path = path / "probs.npy"

        topics = np.load(topics_path)
        probs = np.load(probs_path)

        # TODO: better handling if the model is not found
        if WITH_GPU_SUPPORT:
            topic_model = BERTopic.load(model_path)
        else:
            topic_model = None

        # read and parse the path/model/topics.json file
        # topic_metadata_path = model_path / "topics.json"
        # with open(topic_metadata_path) as f:
        #     topics_data = json.load(f)

        return topic_model, topics, probs

    def save(self):
        
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        
        model_path = self.cache_folder / "model"
        topics_path = self.cache_folder / "topics.npy"
        probs_path = self.cache_folder / "probs.npy"

        topic_model, topics, probs = self.obj

        topic_model.save(
            model_path,
            serialization="safetensors",
            save_embedding_model=True,
            save_ctfidf=True,
        )
        np.save(topics_path, topics)
        np.save(probs_path, probs)


    def get_gpt_representation_model(self, api_key):
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

        prompt = """
                I have topic that contains the following documents: \n[DOCUMENTS]
                The topic is described by the following keywords: [KEYWORDS]

                Based on the above information, can you give a short label of the topic?
                The label should be at most 3 words long and the words should be connected by
                underscores. For example, if the topic is about machine learning the label could
                be "machine_learning". Even if the original text is in a different language,
                the label should be in English.

                Reply only with the label and nothing else.
                """

        client = openai.OpenAI(api_key=api_key)
        gpt_representation = OpenAI(
            client,
            model="gpt-4o-mini",
            prompt=prompt,
            delay_in_seconds=2,
            chat=True,
            nr_docs=4,
            doc_length=100,
            tokenizer=tokenizer,
        )

        return gpt_representation
    
    
    @property
    def cache_folder(self):
        folder = get_cache_folder()
        return folder / "topics" / self.folder_tag