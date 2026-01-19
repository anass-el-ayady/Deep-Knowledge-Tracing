"""
Usage:
    run.py --hidden=<h> [options]

Options:
    --lstm                          Use the LSTM model
    --rnn                           Use the RNN model
    --sakt                          Use the SAKT model
    --fcn                           Use the FCN model
    --dkvmn                         Use the DKVMN model
    --dktplus                       Use the DKT+ model
    --kqn                           Use the KQN model
    --lite                          Use the LITE model
    --lr=<float>                    Learning rate [default: 0.001]
    --bs=<int>                      Batch size [default: 64]
    --seed=<int>                    Seed for reproducibility [default: 59]
    --epochs=<int>                  Number of training epochs [default: 10]
    --cuda=<int>                    GPU identifier to use [default: 0]
    --hidden=<int>                  Hidden state size [default: 128]
    --layers=<int>                  Number of layers [default: 1]
    --heads=<int>                   Number of heads for SAKT [default: 8]
    --dropout=<float>               Dropout rate [default: 0.1]
    --model=<string>                Model type
"""

import os
import random
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from docopt import docopt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import getDataLoader
from evaluation import train_epoch, test_epoch, lossFunc

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_question_ids(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    q_ids = []
    for i in range(0, len(lines), 3):
        try:
            n_steps = int(lines[i].strip())
            question_ids = [int(qid) for qid in lines[i + 1].strip().split(',') if qid.strip()]
            q_ids.append(question_ids[:n_steps])
        except ValueError as e:
            print(f"Error parsing lines {i}-{i+2}: {e}")
    return q_ids

def main():
    args = docopt(__doc__)
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])
    heads = int(args['--heads'])
    dropout = float(args['--dropout'])

    if args['--rnn']:
        model_type = 'RNN'
    elif args['--sakt']:
        model_type = 'SAKT'
    elif args['--lstm']:
        model_type = 'LSTM'
    elif args['--fcn']:
        model_type = 'FCN'
    elif args['--dkvmn']:
        model_type = 'DKVMN'
    elif args['--dktplus']:
        model_type = 'DKTPlus'
    elif args['--kqn']:
        model_type = 'KQN'
    elif args['--lite']:
        model_type = 'LITE'

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    results = []
    base_dir = "dataset"

    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.isdir(dataset_path):
            print(f"\nTraining on dataset: {dataset}")

            train_path = os.path.join(dataset_path, "builder_train.csv")
            test_path = os.path.join(dataset_path, "builder_test.csv")

            q_ids = load_question_ids(train_path)
            questions = max(max(seq) for seq in q_ids) + 1
            max_seq_len = max(len(seq) for seq in q_ids)

            if dataset in ('statics', 'assistChall'):
                length = 500
            elif dataset == 'synthetic':
                length = 50
            else:
                length = 100

            if model_type == 'KQN':
                trainLoader, testLoader = getDataLoader(train_path, test_path, bs, questions, length, kqn=True)
            else:
                trainLoader, testLoader = getDataLoader(train_path, test_path, bs, questions, length)

            if model_type == 'RNN':
                from RNNModel import RNNModel
                model = RNNModel(2 * questions, hidden, layers, questions, device)
            elif model_type == 'LSTM':
                from RNNModel import LSTMModel
                model = LSTMModel(2 * questions, hidden, layers, questions, device)
            elif model_type == 'SAKT':
                from SAKTModel import SAKTModel
                model = SAKTModel(heads, length, hidden, questions, dropout)
            elif model_type == 'FCN':
                from FCNModel import FCNModel
                model = FCNModel(2 * questions, hidden, layers, questions, device)
            elif model_type == 'DKVMN':
                from DKVMNModel import DKVMNModel
                model = DKVMNModel(2 * questions, questions, 50, questions, device)
            elif model_type == 'DKTPlus':
                from DKTPlusModel import DKTPlusModel
                model = DKTPlusModel(2 * questions, hidden, layers, questions, device,
                                     lambda_w1=0.5, lambda_w2=0.5, lambda_o=0.5)
            elif model_type == 'KQN':
                from KQNModel import KQN
                model = KQN(n_skills=questions, n_hidden=hidden, n_rnn_hidden=hidden,
                            n_mlp_hidden=hidden, n_rnn_layers=layers, rnn_type='lstm', device=device)

            model_size = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_func = lossFunc(questions, length, device)

            best_auc = 0
            training_time = 0
            inference_time = 0

            for epoch in range(epochs):
                print(f'Epoch: {epoch + 1}')
                start_train = time.time()
                model, optimizer = train_epoch(model, trainLoader, optimizer, loss_func, device, model_type=model_type)
                training_time += (time.time() - start_train)

                auc, acc, f1 = test_epoch(model, testLoader, loss_func, device, model_type=model_type)
                if auc > best_auc:
                    best_auc = auc

            model.eval()
            start_time = time.time()
            auc, acc, f1 = test_epoch(model, testLoader, loss_func, device, model_type=model_type)
            inference_time = time.time() - start_time

            results.append({
                "dataset": dataset,
                "best_auc": best_auc,
                "f1_score": f1,
                "accuracy": acc,
                "model_size": model_size,
                "training_time": training_time,
                "inference_time": inference_time
            })

    os.makedirs("results", exist_ok=True)
    with open(f"results/performance_results_{model_type}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nAll datasets processed. Results saved to performance_results")

if __name__ == '__main__':
    main()
