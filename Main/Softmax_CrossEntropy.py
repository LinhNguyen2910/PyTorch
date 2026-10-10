import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 1.0]])

probability = torch.softmax(logits, dim=1)
# dim=1 tính softmax theo từng hàng (class)
print("Probability:")
print(probability)
print(probability.sum(dim=1))

prediction = torch.argmax(probability, dim=1)
print("Prediction:")
print(prediction)

criterion = nn.CrossEntropyLoss()
target = torch.tensor([0, 1])
loss = criterion(logits, target)
print("Loss:")
print(loss)