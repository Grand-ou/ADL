import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import time
import csv

import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from sklearn.metrics import classification_report, confusion_matrix

from datasets.dataset import SeqTaggingClsDataset
from util.utils import Vocab
from models.model import SeqTagger

SPLITS = ['test']

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/LSTM_10_14_best_epoch19.pt",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="File to save the model result.",
        default="output/slot_test.csv",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epochs", type=int, default=20)

    args = parser.parse_args()
    return args

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)



    data_paths = {split: args.data_dir for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}


    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, args.max_len, mode='test')
        for split, split_data in data.items()
    }
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # print(embeddings.shape)
    
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(embeddings=embeddings, 
                          hidden_size=args.hidden_size, 
                          num_layers=args.num_layers, 
                          dropout=args.dropout, 
                          bidirectional=args.bidirectional, 
                          num_class=9, 
                          input_feature_dim=300)
    
    test(get_device(), model, datasets, args)


def test_epoch(model, device, dataloader, datasets):

    model.eval()
    tags = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader): 
            inputs, test_id = data

            inputs = inputs.to(device)
            model.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            predictions = torch.max(outputs, 1)[1]
            pred_list = ''
            for pred in predictions:
                if pred_list != '':
                    pred_list+=' '
                pred_list+=datasets.idx2label(pred.item())
            tags.append([test_id[0], pred_list])
    return tags
            

def test(device, model, datasets, args):
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size)
    model.to(device)
    model.load_state_dict(torch.load(args.ckpt_dir))
    header = ['id', 'tags']
    tags = test_epoch(model, device, test_loader, datasets['test'])
    with open(args.pred_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(tags)
        

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    args = parse_args()

    main(args)