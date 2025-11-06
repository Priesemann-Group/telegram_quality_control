import numpy as np
import pandas as pd
import dask.dataframe as dd
import toml
import itertools
from pathlib import Path

from urlextract import URLExtract
import tldextract

import ahocorasick

# ======================================================================================
# Extract URLs from text documents.


def extract_urls(doc, url_extractor=None, remove_duplicates=True):
    """
    Find all urls in the text.

    Args:
        doc (str): Input text document from which to extract URLs.
        url_extractor (URLExtract, optional): An instance of URLExtract to use for extracting URLs.
            If None, a new instance will be created. Pass an existing instance to avoid
            re-downloading the list of top-level domains every time!
        remove_duplicates (bool): If true, multiple occurrences of the same URL in the same
            document will be removed. There might still be duplicates across different documents.

    Returns:
        urls (list): A list of extracted URLs from the document.

    Important: the library we use matches by top-level domains, so it will match
    stuff like "example.com", which is technically not a valid URL because it doesn't
    have a "http://" in the beginning.
    """

    if url_extractor is None:
        url_extractor = URLExtract()

    doc = doc.lower()
    urls = url_extractor.find_urls(doc)

    if remove_duplicates:
        # Remove duplicates
        urls = list(set(urls))

    return urls


def batch_extract_urls(docs, remove_duplicates=True):
    """
    Extract URLs from an array of text documents.

    Args:
        docs (array-like): Input text documents from which to extract URLs.
        remove_duplicates (bool): If true, multiple occurrences of the same URL in the same
            document will be removed. There might still be duplicates across different documents.

    Returns:
        urls: extracted URLs from the documents. If `docs` is a pandas Series, it returns
            a DataFrame with one row per URL and message. If `docs` is a list or NumPy array,
            it returns a list of lists where each inner list contains URLs extracted from the
            corresponding document.
    """

    url_extractor = URLExtract()

    # Extract urls from every message. This creates a Series of lists:
    if isinstance(docs, (pd.Series, dd.Series)):
        url_df = docs.map(lambda doc: extract_urls(doc, url_extractor)).rename("url")

        # Transform the Series of lists into a DataFrame with one row per URL and message.
        # The URLs may repeat if they are mentioned in different messages.
        url_df = url_df.explode().dropna()

        # Change the index to an auto-generated one
        url_df = url_df.to_frame()
        url_df = url_df.reset_index()

        return url_df

    else:
        urls = [
            extract_urls(doc, url_extractor=url_extractor, remove_duplicates=remove_duplicates)
            for doc in docs
        ]
        return urls


# ======================================================================================
# Assign reliability ratings to URLs based on a pre-defined table of domain ratings.


def rate_url(url, domain_ratings_df, automaton, blacklist):
    result = {"domain": np.nan, "is_blacklisted": False, "reliability": np.nan}
    result = pd.Series(result)

    if pd.isna(url):
        return result

    domain = tldextract.extract(url).top_domain_under_public_suffix
    result["domain"] = domain
    result["is_blacklisted"] = domain in blacklist

    # check "t.me" explicitly. It's not in the domain reliability table and it makes out
    # a large part of the URLs in the dataset
    if domain == "t.me":
        return result

    # For all domains in domain_ratings_df, check if the domain is a substring of the URL
    for end_index, (df_index, matched_domain) in automaton.iter(url):
        # Now we need to make sure that the match is not a substring of another domain
        # E.g., if "facebook.com" were not in the table, but "book.com" were,
        # we would get a match for "book.com" in "facebook.com"
        if domain not in matched_domain:
            continue

        result["domain"] = matched_domain
        result["is_blacklisted"] = matched_domain in blacklist
        ranking = domain_ratings_df.loc[df_index]["reliability"]
        result["reliability"] = ranking

    return result


def batch_rate_urls(urls, domain_ratings_df, automaton, blacklist, col=None):
    if isinstance(urls, pd.Series):
        rating_df = urls.apply(
            lambda url: rate_url(url, domain_ratings_df, automaton, blacklist),
        )
        return rating_df

    if isinstance(urls, (pd.DataFrame, dd.DataFrame)):
        rating_df = urls.apply(
            lambda url: rate_url(url[col], domain_ratings_df, automaton, blacklist),
            axis='columns',
            result_type="expand",
        )
        return rating_df

    if isinstance(urls, list):
        if all(isinstance(url, list) for url in urls):
            # if urls is a list of lists:
            rating = [
                [rate_url(url, domain_ratings_df, automaton, blacklist) for url in sublist]
                for sublist in urls
            ]
            rating = [pd.DataFrame(sublist) for sublist in rating]
            return rating

        else:
            # If it's a flat list of URLs
            rating = [rate_url(url, domain_ratings_df, automaton, blacklist) for url in urls]
            rating = pd.DataFrame(rating)
            return rating

    else:
        raise TypeError("Unsupported datatype of the urls!")


def load_rating_resources(version="updated"):
    if version not in ["updated", "original"]:
        raise ValueError("Invalid ranking version. Choose 'updated' or 'original'.")

    resource_folder = Path(__file__).parent.parent / "resources"

    rating_path = resource_folder / "reliability" / version / "domain_ratings.csv"

    # rating_path = Path(f"./resources/reliability/{version}/domain_ratings.csv")

    domain_ratings_df = pd.read_csv(rating_path, header=0, usecols=['domain', 'pc1'])
    domain_ratings_df = domain_ratings_df.rename(columns={"pc1": "reliability"})

    # Sort values by decreasing length so that the longest domain names are checked first
    domain_ratings_df = domain_ratings_df.sort_values(
        by="domain", key=lambda series: series.str.len(), ascending=False
    )
    domain_ratings_df = domain_ratings_df.reset_index(drop=True)

    automaton = ahocorasick.Automaton()

    # add all domains to the automaton
    for i, row in domain_ratings_df.iterrows():
        domain = row['domain']
        automaton.add_word(domain, (i, domain))

    # finalize the automaton
    automaton.make_automaton()

    blacklist_path = resource_folder / "domain_blacklist.toml"
    blacklist = toml.load(blacklist_path)
    blacklist = set(itertools.chain.from_iterable(blacklist.values()))

    return domain_ratings_df, automaton, blacklist


if __name__ == "__main__":
    # Example test case for batch_extract_urls
    print("Testing batch_extract_urls function with different input types:")

    # Sample data with URLs
    text_samples = [
        "Check out this website: reuters.com and also visit http://nytimes.com",
        "No URLs in this text",
        "Multiple occurrences of the same URL: https://telegram.org and https://telegram.org",
        "Different URL formats: example.org, https://pandas.pydata.org/, and http://numpy.org",
    ]

    resources = load_rating_resources(version="updated")

    # Test with a pandas Series
    print("\n1. Testing with pandas Series:")
    series_input = pd.Series(text_samples, name="messages")
    series_input.index.name = "message_id"
    series_results = batch_extract_urls(series_input)
    print(series_results)

    series_ratings = batch_rate_urls(series_results['url'], *resources)
    print(series_ratings)

    df_results = batch_rate_urls(series_results, *resources, col='url')
    print(df_results)

    # Test with a NumPy array
    print("\n2. Testing with NumPy array:")
    numpy_input = np.array(text_samples)
    numpy_results = batch_extract_urls(numpy_input)
    print(numpy_results)
    numpy_ratings = batch_rate_urls(numpy_results, *resources)
    print(numpy_ratings)

    # Test with a list
    print("\n3. Testing with a list:")
    list_input = text_samples
    list_results = batch_extract_urls(list_input)
    print(list_results)
    list_ratings = batch_rate_urls(list_results, *resources)
    print(list_ratings)
