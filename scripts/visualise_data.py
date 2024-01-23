"""
Script for visualising camels datasets with PCA or TSNE.
"""
import argparse
import pickle

import altair as alt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.create_llama_encodings import encode_with_messages_format


def main(args):
    counter = 0
    plots = []
    if args.use_tsne:
        reducer = TSNE(n_components=2)
    else:
        reducer = PCA(n_components=2)

    camel_encoded_data = np.load(args.input_data)
    camel_metadata = pickle.load(open(args.input_metadata, "rb"))
    camels = camel_metadata["camels"]
    camel_lengths = camel_metadata["camel_lengths"]
    tokenizer = AutoTokenizer.from_pretrained(
        "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B", use_fast=False
    )

    transformed_data = reducer.fit_transform(np.stack(camel_encoded_data))
    labels, all_tooltip_data = [], []
    for f in camel_lengths:
        labels += [f] * camel_lengths[f]
        metadata = camels[f]
        tooltip_data = [
            tokenizer.decode(encode_with_messages_format(x, tokenizer, 1024)["input_ids"])
            for x in metadata
        ]
        tooltip_data = [" ".join(x.split())[:200] for x in tooltip_data]
        all_tooltip_data += tooltip_data

    df = pd.DataFrame(transformed_data, columns=["x", "y"])
    df["labels"] = labels
    df["all_tooltip_data"] = all_tooltip_data
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(x="x", y="y", color="labels", tooltip=["labels", "all_tooltip_data"])
        .properties(title="All Datasets")
        .interactive()
    )
    plots.append(chart)
    for f in tqdm(camel_lengths):
        df_data = transformed_data[counter : counter + camel_lengths[f]]
        metadata = camels[f]
        counter += camel_lengths[f]
        tooltip_data = [
            tokenizer.decode(
                encode_with_messages_format(x, tokenizer, 200, args.include_reponse)["input_ids"]
            )
            for x in metadata
        ]
        # remove unnecessary spaces and truncate
        tooltip_data = [" ".join(x.split())[:200] for x in tooltip_data]

        df = pd.DataFrame(df_data, columns=["x", "y"])
        df["inputs"] = tooltip_data
        plot = (
            alt.Chart(df)
            .mark_circle()
            .encode(x="x", y="y", tooltip=["inputs"])
            .properties(title=f)
            .interactive()
        )
        plots.append(plot)
    alt.concat(*plots, columns=2).save("camel_visualisation.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", "-d", type=str)
    parser.add_argument("--input_metadata", "-m", type=str)
    parser.add_argument("--use_tsne", action="store_true")
    parser.add_argument("--include_reponse", action="store_true")
    args = parser.parse_args()

    main(args)
