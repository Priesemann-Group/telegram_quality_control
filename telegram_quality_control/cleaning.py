import pandas as pd
import polars as pl
import dask.dataframe as dd
import numpy as np

import re


# Remove newlines
def clean_text(msg: str):
    # Remove email-like patterns
    msg = re.sub(r'\S+@\S+', '', msg)

    # remove unicode codes
    msg = re.sub(r"\\u[0-9A-Fa-f]{4}", " ", msg)

    # remove links
    msg = re.sub(r"http[s]?://\S+", " ", msg)

    # Replace literal \n, \t, \r with actual whitespace
    msg = msg.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')

    # Replace multiple whitespace (spaces, tabs, newlines) with single space
    msg = re.sub(r'\s+', ' ', msg)

    # Strip leading and trailing whitespace
    msg = msg.strip()

    return msg


def batch_clean_text(docs):
    """
    Clean the message content:
    * Remove newlines
    * Remove unicode codes
    * Remove links

    Returns:
        An Array-Like that contains the text and caption of the message.
    """

    # Pandas
    if isinstance(docs, (pd.Series)):
        clean_docs = docs.map(clean_text)
        return clean_docs

    # Dask Series
    # elif isinstance(docs, dd.Series):
    #     clean_docs = docs.map(clean_text)
    #     return clean_docs

    # Polars Series
    elif isinstance(docs, pl.Series):
        clean_docs = docs.map_elements(clean_text)
        return clean_docs

    # Numpy array
    elif isinstance(docs, np.ndarray):
        clean_docs = np.vectorize(clean_text)(docs)
        return clean_docs

    else:
        clean_docs = [clean_text(msg) for msg in docs]
        return clean_docs


if __name__ == "__main__":
    example_docs = [
        "Hello, how are you doing? Here's a nice link: https://lala.com",
        "I am a weird message \n \n \t with some whitespaces",
        "I'm just chilling",
        "I have some emojis 😉😁",
        "I have russian text: текст на русском",
        "I have some chinese text: 中文文本",
    ]

    all_docs = {
        "pd": pd.Series(example_docs),
        # "dd": dd.Series(example_docs), # doesn't work
        "pl": pl.Series(example_docs),
        "np": np.array(example_docs),
        "list": example_docs,
    }

    for key in all_docs.keys():
        print(f"Datatype: {key}, result: ")
        result = batch_clean_text(all_docs[key])
        print(result)
