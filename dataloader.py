import torch
import torch.utils.data as Data
from readdata import DataReader
import numpy as np

def getDataLoader(train_file, test_file, batch_size, num_of_questions, max_step, kqn=False):
    handle = DataReader(train_file, test_file, max_step, num_of_questions)

    if not kqn:
        # Load data
        train_result = handle.getTrainData(kqn=False)
        test_result = handle.getTestData(kqn=False)

        # Case 1 : random vectors → (data, labels)
        if isinstance(train_result, tuple):
            train_data, train_labels = train_result
            test_data, test_labels = test_result

            dtrain = torch.tensor(np.array(train_data), dtype=torch.float32)
            dtrain_labels = torch.tensor(np.array(train_labels), dtype=torch.long)

            dtest = torch.tensor(np.array(test_data), dtype=torch.float32)
            dtest_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

            train_dataset = Data.TensorDataset(dtrain, dtrain_labels)
            test_dataset = Data.TensorDataset(dtest, dtest_labels)

        # Case 2 : one-hot
        else:
            dtrain = torch.tensor(np.array(train_result), dtype=torch.float32)
            dtest = torch.tensor(np.array(test_result), dtype=torch.float32)

            train_dataset = Data.TensorDataset(dtrain)
            test_dataset = Data.TensorDataset(dtest)

        trainLoader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testLoader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return trainLoader, testLoader

    else:
        train_data = handle.getTrainData(kqn=True)
        test_data = handle.getTestData(kqn=True)

        train_dataset = Data.TensorDataset(
            torch.tensor(train_data['in_data'], dtype=torch.float32),
            torch.tensor(train_data['seq_len'], dtype=torch.long),
            torch.tensor(train_data['next_skills'], dtype=torch.float32),
            torch.tensor(train_data['correctness'], dtype=torch.float32),
            torch.tensor(train_data['mask'], dtype=torch.bool)
        )

        test_dataset = Data.TensorDataset(
            torch.tensor(test_data['in_data'], dtype=torch.float32),
            torch.tensor(test_data['seq_len'], dtype=torch.long),
            torch.tensor(test_data['next_skills'], dtype=torch.float32),
            torch.tensor(test_data['correctness'], dtype=torch.float32),
            torch.tensor(test_data['mask'], dtype=torch.bool)
        )

        trainLoader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testLoader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return trainLoader, testLoader
