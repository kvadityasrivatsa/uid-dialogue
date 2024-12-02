import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import time
import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from tqdm import tqdm
from transformers import RobertaTokenizer, GPT2Tokenizer, RobertaForMaskedLM, GPT2LMHeadModel, AutoTokenizer, \
    AutoModelForCausalLM, TransfoXLTokenizer, TransfoXLLMHeadModel, T5Tokenizer, T5ForConditionalGeneration
from lm_utils import pad, MaptaskSentenceDataset
from pb_processor import Log

logger = logging.getLogger(__name__)

def compute_t5_scores(args, dataframe):
    """
    Compute scores for a fine-tuned T5 model using log probabilities.

    :param args: argparse arguments.
    :param dataframe: a pandas dataframe containing the column 'sentence'.
    """
    print(">> Initializing T5 model and tokenizer...")

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    dataframe["t5_input"] = dataframe["sentence"]
    inputs = tokenizer(
        dataframe["t5_input"].tolist(),
        padding="longest",
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    print(">> Calculating log probabilities...")
    log_2 = torch.log(torch.tensor(2.0))
    sentence_log_probs = []
    sentence_lengths = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(input_ids)), desc="Scoring sentences"):
            input_id = input_ids[i].unsqueeze(0)
            attention = attention_mask[i].unsqueeze(0)

            outputs = model(input_ids=input_id, attention_mask=attention, labels=input_id)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1) / log_2

            target_ids = input_id.squeeze()
            token_log_probs = torch.gather(log_probs.squeeze(), dim=1, index=target_ids.unsqueeze(1)).squeeze()

            valid_token_mask = target_ids != tokenizer.pad_token_id
            token_log_probs = token_log_probs[valid_token_mask]

            sentence_log_probs.append(-token_log_probs.sum().item())
            sentence_lengths.append(valid_token_mask.sum().item())

    dataframe["h"] = sentence_log_probs
    dataframe["length"] = sentence_lengths
    dataframe["normalised_h"] = dataframe["h"] / dataframe["length"]

    print(">> Normalizing scores...")
    h_bar = dataframe.groupby("length").agg({"normalised_h": "mean"})
    xu_h = []
    for _, row in dataframe.iterrows():
        try:
            normalized_score = row["normalised_h"] / h_bar.loc[row["length"], "normalised_h"]
            xu_h.append(normalized_score)
        except KeyError:
            xu_h.append(np.nan)

    dataframe["xu_h"] = xu_h

    out_file_name = os.path.join(
        args.out_path,
        f"{args.model_path.replace('/', '-')}_rc{args.right_context}_ms{args.max_seq_len}.csv"
    )
    dataframe.to_csv(out_file_name, index=False)
    print(f">> Scoring completed. Results saved to: {out_file_name}")
    return out_file_name

def compute_entropy(args, dataframe):
    """
    Compute entropy values for a set of sentences.

    :param args: the argparse script arguments
    :param dataframe: a pandas dataframe containing the column 'sentence'
    """
    # ... 
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="The input data directory."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output data path (for a .txt file)."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="The language model name: 'gpt2', 'roberta', 'dialogpt', 'transfo-xl', or 't5'."
    )
    parser.add_argument(
        "--model_path", type=str, required=False,
        help="The directory path of a trained language model."
    )
    parser.add_argument(
        "--right_context", default=-1, type=int,
        help="The size of the right context window for a bidirectional language model. -1 for the entire context."
    )
    parser.add_argument(
        "--max_seq_len", default=40, type=int,
        help="The maximum number of input tokens."
    )
    parser.add_argument(
        "--add_speaker_ids", action='store_true',
        help="Whether to prepend utterances with speaker identifiers (e.g. 'A: yeah')"
    )
    parser.add_argument(
        "--skip_speaker_ids", action='store_true',
        help="Whether to skip computing the entropy for speaker identifier tokens."
    )
    parser.add_argument(
        "--per_gpu_batch_size", default=4, type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="For distributed training: local_rank."
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(args.data_path)
    dataframe['text'] = dataframe['sentence']

    from datasets import Dataset

    dataset = Dataset.from_pandas(dataframe)
    split = dataset.train_test_split(test_size=0.3, seed=42)
    dataframe = split['test'].to_pandas()

    if args.model_name.lower() == "t5":
        out_file_name = compute_t5_scores(args, dataframe)
    else:
        out_file_name = compute_entropy(args, dataframe)

    logger.warning('Output: {}.csv'.format(out_file_name))
