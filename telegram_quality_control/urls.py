import numpy as np
import pandas as pd
import polars as pl
import dask.dataframe as dd
import toml
import itertools
from pathlib import Path

from urlextract import URLExtract
import tldextract

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

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

# There are two ways on how to tackle this:
# 1. Extract the domain from the URL using URL parsing libraries and then search for this domain
# in the dataset
# 2. Do not use any information from the URL structure. Instead, for every URL, find all domains
# that are substrings of the URL

# The problem with the first approach is that there is such a zoo of subdomains and
# top-level-domains. For example, there's reuters.com, but also pictures.reuters.com,
# graphics.reuters.com and uk.reuters.com, there's bbc.com, but also bbc.co.uk...
# Furthermore, a few Facebook groups are mentioned explicitly in the Lasser dataset.
# For example, "facebook.com/sadefenza" is in the dataset even though it's not a domain.

# The problem with the second approach is that some domains are substrings of other domains. For
# example, "al.com" is in the dataset and will match on "truthsocial.com", which is not in the dataset.

# **What we do:** Use the second approach, but for every successful match extract the domain from
# the URL and make sure that the domain is a substring of the match.


def rate_url(
    url: str,
    domain_ratings_df: pl.DataFrame | pd.DataFrame,
    automaton: ahocorasick.Automaton,
    blacklist: list,
    extended_result=False,
):
    """Rate a single URL based on the domain ratings table and the blacklist.

    Args:
        url (str): The URL to rate.
        domain_ratings_df (pl.DataFrame | pd.DataFrame): The DataFrame containing domain ratings.
        automaton (ahocorasick.Automaton): The automaton for domain matching.
        blacklist (list): A list of blacklisted domains.
        extended_result (bool, optional): if additional fields are awailable in the domain ratings table, include them in the result. Defaults to False.

    Returns:
        _type_: _description_
    """
    result = {"domain": np.nan, "is_blacklisted": False, "reliability": np.nan}
    if extended_result:
        result = {"domain": np.nan, "is_blacklisted": False, "reliability": np.nan}
        for col in domain_ratings_df.columns:
            if col not in result:
                result[col] = np.nan

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
        ranking = domain_ratings_df[df_index, "reliability"]
        result["reliability"] = ranking
        if extended_result:
            for col in domain_ratings_df.columns:
                if col not in ["domain", "reliability"]:
                    result[col] = domain_ratings_df[df_index, col]

    return result


def batch_rate_urls(url_df, url_col="url", version="updated", extended_result=False):
    """
    Rate all URLs in the input DataFrame using the domain ratings table.

    Args:
        urls (DataFrame): Input URLs to be rated.
        version (str): The version of the domain ratings to use. Can be "updated" or "original".
    """
    domain_ratings_df, automaton, blacklist = load_rating_resources(version=version)

    using_dask = isinstance(url_df, dd.DataFrame)
    using_polars = isinstance(url_df, (pl.DataFrame, pl.Series))
    using_pandas = isinstance(url_df, (pd.DataFrame, pd.Series))
    if not (using_dask or using_polars or using_pandas):
        raise ValueError(
            "Input must be a Dask DataFrame, Polars DataFrame/Series, or Pandas DataFrame/Series."
        )

    if using_dask:
        print("Using Dask for parallel processing of URLs.")
        result = url_df.map_partitions(batch_rate_urls, url_col=url_col, version=version, extended_result=extended_result)
        return result

    if isinstance(url_df, (pd.Series, pl.Series)):
        urls = url_df
    else:
        urls = url_df[url_col]

    if using_pandas:
        meta = {"domain": "string", "is_blacklisted": bool, "reliability": "Float64"}
        if extended_result:
            for col in domain_ratings_df.columns:
                if col not in meta.keys():
                    meta[col] = "object"
        result = pd.DataFrame(columns=list(meta.keys()), index=url_df.index)
        result = result.astype(meta)
        result["domain"] = None
        result["is_blacklisted"] = False
        result["reliability"] = np.nan

        for id, url in urls.items():
            row_result = rate_url(url, domain_ratings_df, automaton, blacklist, extended_result=extended_result)
            result.loc[id] = row_result

    elif using_polars:
        meta = {"domain": pl.String, "is_blacklisted": pl.Boolean, "reliability": pl.Float64}
        
        if extended_result:
            for col in domain_ratings_df.columns:
                if col not in meta.keys():
                    meta[col] = pl.Utf8

        all_results = []
        for id, url in enumerate(urls):
            row_result = rate_url(url, domain_ratings_df, automaton, blacklist, extended_result=extended_result)
            all_results.append(row_result)
        result = pl.DataFrame(all_results, schema=meta)

    return result


def load_rating_resources(version="updated"):
    if version not in ["updated", "original"]:
        raise ValueError("Invalid ranking version. Choose 'updated' or 'original'.")

    resource_folder = Path(__file__).parent.parent / "resources"

    rating_path = resource_folder / "reliability" / version / "domain_ratings.csv"

    domain_ratings_df = pl.read_csv(rating_path, has_header=True)

    # Sort values by decreasing length so that the longest domain names are checked first
    domain_ratings_df = domain_ratings_df.sort(pl.col("domain").str.len_chars(), descending=True)

    automaton = ahocorasick.Automaton()

    # add all domains to the automaton
    for i, row in enumerate(domain_ratings_df.iter_rows(named=True)):
        domain = row['domain']
        automaton.add_word(domain, (i, domain))

    # finalize the automaton
    automaton.make_automaton()

    blacklist_path = resource_folder / "domain_blacklist.toml"
    blacklist = toml.load(blacklist_path)
    blacklist = set(itertools.chain.from_iterable(blacklist.values()))

    return domain_ratings_df, automaton, blacklist

