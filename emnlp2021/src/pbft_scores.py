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
    AutoModelForCausalLM, TransfoXLTokenizer, TransfoXLLMHeadModel
# from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, \
#     AutoModelForCausalLM, TransfoXLTokenizer, TransfoXLLMHeadModel
from lm_utils import pad, MaptaskSentenceDataset
from pb_processor import Log

logger = logging.getLogger(__name__)


def load_logs(dir_path):
    print('>> Loading logs from "{}"'.format(dir_path))

    file_count = 0
    for _, _, files in os.walk(dir_path):
        for file in files:
            file_count += int(file.endswith('.json'))
    print('{} files found.'.format(file_count))

    logs = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs[log.game_id] = log

    print('DONE. Loaded {} completed game logs.'.format(len(logs)))
    return logs


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def compute_entropy(args, dataframe):
    """
    Compute entropy values for a set of sentences.

    :param args: the argparse script arguments
    :param dataframe: a pandas dataframe containing the column 'sentence'
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s',
        args.local_rank,
        device,
        n_gpu,
        bool(args.local_rank != -1)
    )

    # Set seeds across modules
    set_seed(args.seed, n_gpu)

    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    if not args.model_path:
        if args.model_name == 'dialogpt':
            args.model_path = 'microsoft/DialoGPT-small'
        elif args.model_name.lower() == 'transfo-xl':
            args.model_path = 'transfo-xl-wt103'
        elif args.model_name.lower() == 'roberta':
            args.model_path = 'roberta-base'
        else:
            args.model_path = args.model_name

    # Load LM and tokenizer
    if args.model_name.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        lm = RobertaForMaskedLM.from_pretrained(args.model_path, return_dict=True)
    elif args.model_name.lower() == 'gpt2':
        # tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        # lm = GPT2LMHeadModel.from_pretrained(args.model_path, return_dict=True)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        lm = GPT2LMHeadModel.from_pretrained(args.model_path, return_dict=True)
        # tokenizer = AutoTokenizer.from_pretrained("./gpt2-pbft_final")
        # tokenizer.pad_token = tokenizer.eos_token(
        # lm = AutoModelForCausalLM.from_pretrained("./gpt2-pbft_final", return_dict=True)
    elif args.model_name.lower() == 'dialogpt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        lm = AutoModelForCausalLM.from_pretrained(args.model_path)
    elif args.model_name.lower() == 'transfo-xl':
        tokenizer = TransfoXLTokenizer.from_pretrained(args.model_path)
        lm = TransfoXLLMHeadModel.from_pretrained(args.model_path)
    # else:
    #     tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    #     lm = RobertaForMaskedLM.from_pretrained(args.model_name)
    #     #raise ValueError('Incorrect model name: {}. Available: gpt2, roberta, and dialogpt'.format(args.model_name))

    lm.to(device)

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training
        # download model & vocab
        torch.distributed.barrier()

    args.batch_size = args.per_gpu_batch_size * max(1, n_gpu)

    if args.model_name == 'transfo-xl':
        def collate(batch):
            return [
                pad(tokenizer, [item[0] for item in batch], attention_mask=False),
                [item[1] for item in batch]
            ]
    else:
        def collate(batch):
            return [
                pad(tokenizer, [item[0] for item in batch]),
                [item[1] for item in batch]
            ]

    data = MaptaskSentenceDataset(dataframe, tokenizer, args.max_seq_len, args.add_speaker_ids)
    sampler = SequentialSampler(data) if args.local_rank == -1 else DistributedSampler(data, shuffle=False)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)

    # multi-gpu
    if n_gpu > 1:
        lm = torch.nn.DataParallel(lm)

    # Distributed
    if args.local_rank != -1:
        lm = torch.nn.parallel.DistributedDataParallel(
            lm, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    log_2 = torch.log(torch.tensor(2.))

    sentence_sumlogp = []
    sentence_length = []
    sentence_idx = []

    if args.skip_speaker_ids:
        speaker_ids = tokenizer.convert_tokens_to_ids(['A', 'B', 'ĠA', 'ĠB'])
        colon_id = tokenizer.convert_tokens_to_ids([':'])[0]

    unk_id = tokenizer.convert_tokens_to_ids('<unk>')

    logger.warning('Compute entropy...')
    iterator = tqdm(dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(iterator):

        inputs, idx = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        try:
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
        except KeyError:
            pass
        batch_sumlogp = [0 for _ in idx]
        batch_lengths = [0 for _ in idx]
        max_sent_len = inputs['input_ids'].shape[1]

        lm.eval()

        if args.model_name.lower() == 'roberta':
            # for every next token...
            for token_index in range(1, max_sent_len):

                # mask next token (for every sentence in the batch)
                masked_inputs = {
                    'input_ids': inputs['input_ids'].clone(),
                    'attention_mask': inputs['attention_mask'].clone()
                }

                masked_inputs['input_ids'][:, token_index] = tokenizer.mask_token_id
                if args.right_context >= 0:
                    masked_inputs['attention_mask'][:, token_index + 1 + args.right_context:] = torch.tensor(0)

                masked_inputs['input_ids'] = masked_inputs['input_ids'].to(device)
                masked_inputs['attention_mask'] = masked_inputs['attention_mask'].to(device)

                # run LM to obtain next token probabilities (for every sentence in the batch)
                with torch.no_grad():
                    outputs = lm(**masked_inputs)  # n_sentences, max_sent_len, vocab_size

                # get log probability of next token (for every sentence in the batch)
                logp_w = log_softmax(outputs.logits[:, token_index, :], dim=-1)  # n_sentences, vocab_size
                logp_w /= log_2  # change to base 2

                # for every sentence in the batch...
                for s_id in range(inputs['input_ids'].shape[0]):

                    # get next token id (for this sentence)
                    w_id = inputs['input_ids'][s_id, token_index]

                    # skip special tokens (BOS, EOS, PAD)
                    if w_id in tokenizer.all_special_ids and w_id != unk_id:
                        continue
                    # skip speaker identifier
                    if args.skip_speaker_ids \
                            and w_id in speaker_ids \
                            and inputs['input_ids'][s_id, token_index - 1] == tokenizer.bos_token_id \
                            and inputs['input_ids'][s_id, token_index + 1] == colon_id:
                        continue
                    # skip colon after speaker identifier
                    if args.skip_speaker_ids \
                            and w_id == colon_id \
                            and inputs['input_ids'][s_id, token_index - 1] in speaker_ids \
                            and inputs['input_ids'][s_id, token_index - 2] == tokenizer.bos_token_id:
                        continue

                    # increase sentence length if next token is not special token
                    batch_lengths[s_id] += 1
                    # increase non-normalised log probability of the sentence
                    batch_sumlogp[s_id] += logp_w[s_id, w_id].item()
        else:
            # run unidirectional LM to obtain next token probabilities (for every sentence in the batch)
            with torch.no_grad():
                outputs = lm(**inputs)  # n_sentences, max_sent_len, vocab_size

            logp_w = log_softmax(outputs.logits, dim=-1)
            logp_w /= log_2

            # for every token...
            for token_index in range(max_sent_len - 1):

                # get next token id (for every sentence in the batch)
                w_ids = inputs['input_ids'][:, token_index + 1]  # n_sentences

                # for every sentence in the batch...
                for s_id in range(inputs['input_ids'].shape[0]):

                    # get next token id (for this sentence)
                    w_id = w_ids[s_id]

                    # skip special tokens (BOS, EOS, PAD)
                    if w_id in tokenizer.all_special_ids and (w_id != unk_id or args.model_name == 'gpt2'):
                        continue
                    # skip speaker identifier
                    if args.skip_speaker_ids \
                            and w_id in speaker_ids \
                            and inputs['input_ids'][s_id, token_index] == tokenizer.eos_token_id \
                            and inputs['input_ids'][s_id, token_index + 2] == colon_id:
                        continue
                    # skip colon after speaker identifier
                    if args.skip_speaker_ids \
                            and w_id == colon_id \
                            and inputs['input_ids'][s_id, token_index] in speaker_ids \
                            and inputs['input_ids'][s_id, token_index - 1] == tokenizer.eos_token_id:
                        continue

                    # increase sentence length if next token is not special token
                    batch_lengths[s_id] += 1
                    # increase non-normalised log probability of the sentence
                    batch_sumlogp[s_id] += logp_w[s_id, token_index, w_id].item()

        sentence_sumlogp.extend(batch_sumlogp)
        sentence_length.extend(batch_lengths)
        sentence_idx.extend(idx)

    iterator.close()
    logger.warning('--- %s seconds ---' % (time.time() - start_time))

    sentence_sumlogp = - np.array(sentence_sumlogp)
    sentence_length = np.array(sentence_length)
    sentence_avglogp = sentence_sumlogp / sentence_length
    sentence_idx = np.array(sentence_idx)

    dataframe.loc[:, 'h'] = np.nan
    dataframe.loc[:, 'normalised_h'] = np.nan
    dataframe.loc[:, 'length'] = np.nan

    for idx, h, n_h, len in zip(sentence_idx, sentence_sumlogp, sentence_avglogp, sentence_length):
        dataframe.loc[idx, 'h'] = h
        dataframe.loc[idx, 'normalised_h'] = n_h
        dataframe.loc[idx, 'length'] = len

    h_bar = dataframe.groupby('length').agg({"normalised_h": "mean"})
    xu_h = []
    for index, row in dataframe.iterrows():
        try:
            xu_h.append(row['normalised_h'] / h_bar.loc[row['length'], 'normalised_h'])
        except KeyError:
            xu_h.append(np.nan)

    dataframe.loc[:, 'xu_h'] = xu_h

    out_file_name = os.path.join(args.out_path, '{}_{}{}{}_{}'.format(
        args.model_path.replace('/', '-'),
        'skip_' if args.skip_speaker_ids else '',
        'ids_' if args.add_speaker_ids else '',
        args.right_context,
        args.max_seq_len
    ))
    dataframe.to_csv(
        '{}.csv'.format(out_file_name),
        index=False,
    )

    return out_file_name


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
        help="The language model name: 'gpt2', 'roberta', 'dialogpt' or 'transfo-xl'."
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

    out_file_name = compute_entropy(args, dataframe)
    logger.warning('Output: {}.csv'.format(out_file_name))
