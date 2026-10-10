import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyData(Dataset):
    def __init__(self, file, transform=None):
        xy = np.loadtxt(file, delimiter=",", dtype=np.float32, skiprows=1)
        # delimiter = "," ký tự dùng để ngăn cách các cột trong file
        # skiprows = 1 bỏ dòng tiêu đề (dòng 1)

        self.y = torch.from_numpy(xy[:, [0]])
        # Lấy cột đầu tiên là

        self.x = torch.from_numpy(xy[:, 1:])
        # Các cột còn lại là features

        self.transform = transform
        # Save transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Lấy một sample tại vị trí index
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)

        return x, y

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

data = np.loadtxt("../Data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
X = data[:, 1:]

mean = torch.tensor(X.mean(axis=0), dtype=torch.float32)
# Tính mean của từng feature

std = torch.tensor(X.std(axis=0), dtype=torch.float32)
# Tính std của từng feature

std[std == 0] = 1 # Tránh trường hợp std = 0

transform = Normalize(mean=mean, std=std)

dataset = MyData(file="../Data/wine.csv", transform=transform)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
# Với Windows/Pycharm có thể đẻe 0 khi mới học chỉ số num_workers
# drop_last=False là không bỏ batch cuối

for x_batch, y_batch in train_loader:
    print("\nBatch:")
    print("X batch:", x_batch.shape)
    print("Y batch:", y_batch.shape)

    print("\nX batch:")
    print(x_batch)

    print("\nY batch:")
    print(y_batch)

    break
