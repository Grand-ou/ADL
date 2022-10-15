import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader


from datasets.dataset import SeqClsDataset
from util.utils import Vocab
from models.model import SeqClassifier

SPLITS = ['test']

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()
    return args

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)



    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    # k = [split for split, split_data in data.items()]
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, args.max_len, mode='test')
        for split, split_data in data.items()
    }
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # print(embeddings.shape)
    
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, 
                          hidden_size=args.hidden_size, 
                          num_layers=args.num_layers, 
                          dropout=args.dropout, 
                          bidirectional=args.bidirectional, 
                          num_class=150, 
                          input_feature_dim=300)

    test(get_device(), model, datasets, args)

    # TODO: Inference on test set

def test_epoch(model, device, dataloader, datasets):

    model.eval()
    intents = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader): 
            inputs, test_id = data

            inputs = inputs.to(device)
            model.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            predictions = torch.max(outputs, 1)[1]
            for pred, id in zip(predictions, test_id):
                intents.append([id, datasets.idx2label(pred.item())])
    return intents
def test(device, model, datasets, args):
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, collate_fn=datasets['test'].collate_fn)
    model.to(device)
    model.load_state_dict(torch.load(args.ckpt_dir))
    header = ['id', 'intent']
    intents = test_epoch(model, device, test_loader, datasets['test'])
    with open('output/intent_test.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(intents)
        
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    args = parse_args()
    main(args)
