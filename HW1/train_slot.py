import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import time

import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix

from datasets.dataset import SeqTaggingClsDataset
from util.utils import Vocab
from models.model import SeqTagger

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
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
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='initial momentum')
    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epochs", type=int, default=30)

    args = parser.parse_args()
    return args

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)



    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}


    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, args.max_len)
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

    # TODO: init optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    train(get_device(), model, datasets, args, optimizer)


    # TODO: Inference on test set



def train_epoch(model, device, dataloader, criterion, optimizer):
    train_loss,train_correct = 0.0, 0
    model.train()
    for data in tqdm.tqdm(dataloader):
        
        inputs, labels = data
        # print('input = ', inputs)
        # print('input = ', inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        if len(labels.shape) > 1:
            labels = labels.squeeze(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        predictions = torch.max(outputs, 1)[1]
        train_loss += loss.item()
        correct = 0
        if torch.equal(predictions, labels):
            correct = 1 

        train_correct += correct
    return train_loss, train_correct

def valid_epoch(model, device, dataloader, criterion):
    y_true = []
    y_pred = []
    train_loss, train_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            
            inputs, labels = data
            
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(inputs)
            if len(labels.shape) > 1:
                labels = labels.squeeze(0)
            loss = criterion(outputs, labels)
            predictions = torch.max(outputs, 1)[1]
            
            y_true.append(labels.cpu().tolist())
            y_pred.append(predictions.cpu().tolist())
            correct = 0
            if torch.equal(predictions, labels):
                correct = 1 
            train_loss += loss.item()
            train_correct += correct
        # print(y_true)
    return y_true, y_pred, train_loss, train_correct

def train(device, model, datasets, args, optimizer):
    criterion = nn.CrossEntropyLoss() 
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size)
    test_loader = DataLoader(datasets['eval'], batch_size=args.batch_size)
    # print(train_loader[0])
    model.to(device)
    highest_test_acc  = 0.0
    for epoch in range(args.num_epochs):
        train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
        true, pred, test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler)
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler)

        print("  Epoch:{}/{}".format(epoch + 1, args.num_epochs))                                               
        print("    Training Loss:{:.4f},  Training Acc {:.2f}".format(train_loss, train_acc))
        print("    Testing Loss:{:.4f},  Testing Acc {:.2f}".format(test_loss, test_acc))
        if test_acc > highest_test_acc:
            best_epoch_num = epoch+1
            # save_model(best_epoch_num, model, args)
            highest_test_acc = test_acc
        


    print(classification_report(true, pred))
    # draw_curve(args, average_history)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def save_model(epoch_num, model, args):
    print('Best epoch: {}'.format(epoch_num))
    print('Saving model...')
    result = time.localtime(time.time())
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'LSTM_'+str(result.tm_mon)+'_'+str(result.tm_mday)+'_best_epoch'+str(epoch_num)+'.pt'))
    print('Model saved.')

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)