import numpy as np
import itertools
from tqdm import tqdm

class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques, embed_dim=16, threshold=1000):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques
        self.embed_dim = embed_dim
        self.threshold = threshold

    def getData(self, file_path, return_kqn_format=False):
        data = []
        # Only used if random vector is enabled
        use_random_vectors = self.numofques > self.threshold
        true_labels = [] if use_random_vectors else None

        # KQN-specific
        in_data_list = []
        next_skills_list = []
        correctness_list = []
        seq_len_list = []
        mask_list = []

        if use_random_vectors:
            print(f"Using random vector projection (2M={2 * self.numofques}) -> embed_dim={self.embed_dim}")
            n_vectors = np.random.normal(0, 1, size=(2 * self.numofques, self.embed_dim))
        else:
            print(f"Using standard one-hot encoding (2M={2 * self.numofques})")

        # First, count number of records
        with open(file_path, 'r') as file:
            num_lines = sum(1 for _ in file)
        num_records = num_lines // 3

        with open(file_path, 'r') as file:
            for len_line, ques_line, ans_line in tqdm(itertools.zip_longest(*[file] * 3), total=num_records, desc="Processing records"):
                length = int(len_line.strip().strip(','))
                questions = [int(q) for q in ques_line.strip().strip(',').split(',')]
                answers = [int(a) for a in ans_line.strip().strip(',').split(',')]

                slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)

                for i in range(slices):
                    start = i * self.maxstep
                    end = min(start + self.maxstep, length)
                    steps = end - start

                    # Allocate interaction vector
                    if use_random_vectors:
                        interaction = np.zeros((self.maxstep, self.embed_dim))
                    else:
                        interaction = np.zeros((self.maxstep, 2 * self.numofques))

                    if return_kqn_format:
                        next_skills = np.zeros((self.maxstep, self.numofques))
                        correctness = np.zeros((self.maxstep,))
                        mask = np.zeros((self.maxstep,), dtype=bool)

                    for j in range(steps):
                        q = questions[start + j]
                        a = answers[start + j]
                        interaction_id = q if a == 1 else q + self.numofques

                        if use_random_vectors:
                            interaction[j] = n_vectors[interaction_id]
                        else:
                            interaction[j][interaction_id] = 1

                        if return_kqn_format and j < steps - 1:
                            q_next = questions[start + j + 1]
                            a_next = answers[start + j + 1]
                            next_skills[j][q_next] = 1
                            correctness[j] = a_next
                            mask[j] = True

                    data.append(interaction.tolist())

                    if use_random_vectors:
                        true_labels.append(answers[start:end])

                    if return_kqn_format:
                        in_data_list.append(interaction)
                        next_skills_list.append(next_skills)
                        correctness_list.append(correctness)
                        mask_list.append(mask)
                        seq_len_list.append(steps)

        print('done:', np.array(data).shape)

        if return_kqn_format:
            return {
                'in_data': np.array(in_data_list),
                'next_skills': np.array(next_skills_list),
                'correctness': np.array(correctness_list),
                'mask': np.array(mask_list),
                'seq_len': np.array(seq_len_list)
            }

        if use_random_vectors:
            max_len = max(len(x) for x in true_labels)
            true_labels = [x + [0] * (max_len - len(x)) for x in true_labels]
            return np.array(data), np.array(true_labels)
        else:
            return np.array(data)

    def getTrainData(self, kqn=False):
        print('loading train data...')
        return self.getData(self.train_path, return_kqn_format=kqn)

    def getTestData(self, kqn=False):
        print('loading test data...')
        return self.getData(self.test_path, return_kqn_format=kqn)

