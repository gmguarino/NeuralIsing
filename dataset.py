import torch
import os


class Ising(torch.utils.data.Dataset):
    
    def __init__(self, data_path, transforms=None, train=True):
        super(Ising, self).__init__()
        self.root = data_path
        if train:
            self.path = data_path + "/train"
            self.transforms = transforms
        else:
            self.path = data_path + "/test"
            self.transforms = None
        self.files = list(os.listdir(self.path))

    def __getitem__(self, index):
        file_path = os.path.join(self.path, self.files[index])
        target = dict()
        # Order: Mat, T, E, M, size
        with open(file_path, 'r') as f:
            lines = f.read()
            lines = lines.split("\n")[:-1]
        target["T"] = torch.tensor(float(lines[1]), dtype=torch.float32)
        target["E"] = torch.tensor(float(lines[2]), dtype=torch.float32)
        target["M"] = torch.tensor(float(lines[3]), dtype=torch.float32)
        target["size"] = int(lines[4])
        matrix = lines[0].split(" ")[:-1]
        matrix = list(map(float, matrix))
        matrix = torch.tensor(matrix, dtype=torch.float32)
        matrix = matrix.view(target["size"], target["size"]).unsqueeze(0)
        if self.transforms:
            matrix = self.transforms(matrix)

        return matrix, target

    def __len__(self):
        return len(self.files)
