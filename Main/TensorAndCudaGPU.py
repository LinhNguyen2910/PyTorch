import torch


# ================================================================
# 1. DEVICE - CHỌN CPU / GPU
# Kiểm tra máy có CUDA/GPU NVIDIA mà PyTorch sử dụng được hay không
print(torch.cuda.is_available())
# Tự động chọn GPU nếu có, nếu không thì dùng CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)
# cuda  -> đang sử dụng GPU
# cpu   -> đang sử dụng CPU


# ================================================================
# 2. THÔNG TIN GPU
if torch.cuda.is_available():
    # Số lượng GPU
    print("GPU count:", torch.cuda.device_count())
    # Tên GPU thứ 0
    print("GPU name:", torch.cuda.get_device_name(0))


# ================================================================
# 3. TẠO TENSOR
# Tạo Tensor từ list
x = torch.tensor([1, 2, 3])
# Tensor toàn số 0
x = torch.zeros(2, 3)
# Tensor toàn số 1
x = torch.ones(2, 3)
# Tensor random trong khoảng [0, 1)
x = torch.rand(2, 3)
# Tensor random theo phân phối chuẩn N(0, 1)
x = torch.randn(2, 3)
# Tạo dãy số: 0, 2, 4, 6, 8
x = torch.arange(0, 10, 2)
# Ma trận đơn vị
x = torch.eye(3)


# ================================================================
# 4. KIỂM TRA TENSOR
x = torch.randn(2, 3, 4)
print(x.shape)
# Kích thước Tensor
# torch.Size([2, 3, 4])
print(x.ndim)
# Số chiều của Tensor
# 3
print(x.numel())
# Tổng số phần tử
# 2 * 3 * 4 = 24
print(x.dtype)
# Kiểu dữ liệu, thường là torch.float32
print(x.device)
# Tensor đang nằm ở CPU hay GPU


# ================================================================
# 5. DTYPE - KIỂU DỮ LIỆU
x = torch.tensor([1, 2, 3])
x = x.float()       # Chuyển sang float32
x = x.double()      # Chuyển sang float64
x = x.long()        # Chuyển sang int64
# Hoặc dùng .to()
x = x.to(torch.float32)


# ================================================================
# 6. INDEXING / SLICING
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(x[0])
# Lấy hàng đầu tiên
# tensor([1, 2, 3])
print(x[1])
# Lấy hàng thứ hai
print(x[:, 0])
# Lấy cột đầu tiên
# tensor([1, 4])
print(x[0, 1])
# Lấy phần tử hàng 0, cột 1
# tensor(2)
print(x[0:2])
# Lấy các hàng từ 0 đến 1


# ================================================================
# 7. THAY ĐỔI SHAPE
x = torch.randn(2, 3, 4)
# reshape: thay đổi shape nhưng giữ nguyên số phần tử
x = x.reshape(6, 4)
# view: tương tự reshape trong nhiều trường hợp
x = x.view(6, 4)
# flatten: làm phẳng Tensor
x = x.flatten()
x = torch.randn(32, 3, 28, 28)
# 32 = batch size
# 3  = số channel
# 28 = height
# 28 = width
x = x.flatten(1)
# Kết quả:
# (32, 3, 28, 28)
#       ↓
# (32, 2352)
#
# Giữ dimension 0 (batch), gộp các dimension còn lại.


# ================================================================
# 8. UNSQUEEZE / SQUEEZE
x = torch.tensor([1, 2, 3])
# Thêm một dimension
x = x.unsqueeze(0)
# (3,) → (1, 3)
# Bỏ dimension có kích thước = 1
x = x.squeeze()
# (1, 3) → (3,)
# Ví dụ ảnh:
image = torch.randn(3, 224, 224)
image = image.unsqueeze(0)
# (3, 224, 224)
#       ↓
# (1, 3, 224, 224)
#
# Thêm batch dimension.


# ================================================================
# 9. CÁC PHÉP TOÁN CƠ BẢN
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
a + b       # Cộng từng phần tử
a - b       # Trừ từng phần tử
a * b       # Nhân từng phần tử
a / b       # Chia từng phần tử
a ** 2      # Bình phương từng phần tử


# ================================================================
# 10. MATRIX MULTIPLICATION
A = torch.randn(2, 3)
B = torch.randn(3, 4)
# Nhân ma trận
C = A @ B
# Hoặc:
C = torch.matmul(A, B)
# (2, 3) @ (3, 4)
#       ↓
#     (2, 4)
# LƯU Ý:
# A * B  → nhân từng phần tử (element-wise)
# A @ B  → nhân ma trận (matrix multiplication)


# ================================================================
# 11. CÁC HÀM TÍNH TOÁN
x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])
x.sum()          # Tổng tất cả phần tử
x.mean()         # Trung bình
x.max()          # Giá trị lớn nhất
x.min()          # Giá trị nhỏ nhất
x.argmax()       # Vị trí của phần tử lớn nhất
x.argmin()       # Vị trí của phần tử nhỏ nhất


# ================================================================
# 12. DIM - CHIỀU TÍNH TOÁN
x.sum(dim=0)
# Cộng theo dimension 0
#
# [[1, 2, 3],
#  [4, 5, 6]]
#
# → [5, 7, 9]
x.sum(dim=1)
# Cộng theo dimension 1
#
# → [6, 15]


# ================================================================
# 13. CAT / STACK
a = torch.tensor([[1, 2]])
b = torch.tensor([[3, 4]])
# CAT:
# Nối Tensor vào dimension đã tồn tại
x = torch.cat([a, b], dim=0)
# [[1, 2],
#  [3, 4]]
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
# STACK:
# Tạo thêm một dimension mới
x = torch.stack([a, b])
# [[1, 2],
#  [3, 4]]
# Nhớ:
# cat   → nối vào dimension có sẵn
# stack → tạo dimension mới


# ================================================================
# 14. NUMPY ↔ TENSOR
import numpy as np
array = np.array([1, 2, 3])
# NumPy → Tensor
x = torch.from_numpy(array)
# Hoặc:
x = torch.tensor(array)
# Tensor → NumPy
array = x.numpy()
# Nếu Tensor đang ở GPU:
array = x.cpu().numpy()
# GPU → CPU → NumPy


# ================================================================
# 15. ĐƯA TENSOR LÊN GPU
x = torch.randn(3, 4)
# CPU → device
x = x.to(device)
# Nếu device = cuda thì x đang nằm trên GPU
# Có thể tạo trực tiếp trên GPU:
x = torch.randn(
    3,
    4,
    device=device
)
# Hoặc:
x = torch.randn(3, 4).cuda()
# Tuy nhiên nên ưu tiên .to(device)
# vì code chạy được cả CPU và GPU.


# ================================================================
# 16. GPU → CPU
x = x.cpu()
# Hoặc:
x = x.to("cpu")


# ================================================================
# 17. KIỂM TRA DEVICE
print(x.device)
# cpu    → Tensor đang ở CPU
# cuda:0 → Tensor đang ở GPU số 0


# ================================================================
# 18. QUY TẮC DEVICE
a = torch.randn(3).to(device)
b = torch.randn(3).to(device)
c = a + b
# Đúng:
# a và b cùng device
# Sai:
# a ở GPU
# b ở CPU
# a + b → ERROR
# Trong Neural Network:
# model → GPU
# input → GPU
# label → GPU


# ================================================================
# 19. AUTOGRAD
# requires_grad=True:
# PyTorch bắt đầu theo dõi phép tính của Tensor này.
x = torch.tensor(
    3.0,
    requires_grad=True
)
y = x ** 2
# Tính đạo hàm dy/dx
y.backward()
# Lấy gradient
print(x.grad)
# y = x²
# dy/dx = 2x
# x = 3
# → gradient = 6


# ================================================================
# 20. ZERO GRADIENT
x.grad.zero_()
# Xóa gradient cũ.
#
# Trong PyTorch gradient mặc định được cộng dồn,
# vì vậy khi training cần xóa gradient của batch trước.


# ================================================================
# 21. DETACH
x = torch.tensor(
    3.0,
    requires_grad=True
)
y = x ** 2
z = y.detach()
# detach():
# Tạo Tensor có cùng giá trị nhưng tách khỏi
# computation graph của Autograd.


# ================================================================
# 22. NO_GRAD
with torch.no_grad():
    y = x ** 2
# Không theo dõi gradient trong block này.
#
# Thường dùng khi:
# - Test model
# - Inference
# - Không cần Backpropagation
#
# Giúp tiết kiệm RAM / VRAM.


# ================================================================
# 23. MODEL + CUDA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Định nghĩa thiết bị (Sử dụng GPU nếu có, không thì dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")
# 2. Tạo một mô hình đơn giản để test (ví dụ mạng Linear)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)  # Đầu vào 10 chiều, đầu ra 2 lớp
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
# 3. Đưa toàn bộ model lên GPU
model = model.to(device)
# 4. Tạo dữ liệu giả lập (Dummy Data) và DataLoader
X_data = torch.randn(100, 10)  # 100 mẫu, mỗi mẫu 10 đặc trưng
y_data = torch.randint(0, 2, (100,))  # Nhãn phân loại 0 hoặc 1
dataset = TensorDataset(X_data, y_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# 5. Khởi tạo Hàm mất mát (Loss function) và Bộ tối ưu hóa (Optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 6. Vòng lặp huấn luyện (Training Loop)
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Đặt model ở chế độ training
    running_loss = 0.0
    for X, y in dataloader:
        # Đưa dữ liệu lên GPU
        X = X.to(device)
        y = y.to(device)
        # Xóa gradient cũ
        optimizer.zero_grad()
        # Forward (Lan truyền tiến)
        output = model(X)
        # Tính Loss
        loss = criterion(output, y)
        # Backpropagation (Lan truyền ngược)
        loss.backward()
        # Cập nhật parameters (Trọng số)
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# ================================================================
# 24. TRAIN / EVAL
# Chế độ Training
model.train()
# Chế độ Evaluation / Testing
model.eval()
# Khi test thường dùng kết hợp:
model.eval()
with torch.no_grad():
    output = model(X)


# ================================================================
# 25. CUDA MEMORY
if torch.cuda.is_available():
    # VRAM PyTorch đang thực sự sử dụng
    print(torch.cuda.memory_allocated())
    # VRAM PyTorch đã reserve
    print(torch.cuda.memory_reserved())
    # Xóa cache không còn sử dụng
    torch.cuda.empty_cache()


# ================================================================
# 26. CUDA SYNCHRONIZE
torch.cuda.synchronize()
# CUDA thường thực hiện phép tính bất đồng bộ.
# synchronize() buộc CPU chờ GPU hoàn thành.
#
# Thường dùng khi benchmark thời gian chạy GPU.


# ================================================================
#                    TRAINING PIPELINE
# ================================================================

"""
                  CPU
                   │
                   │ .to(device)
                   ▼
                  GPU
                   │
                   ▼
              Tensor X
                   │
                   ▼
              model(X)
                   │
                   ▼
               Prediction
                   │
                   ▼
                 Loss
                   │
             backward()
                   │
                   ▼
               Gradient
                   │
           optimizer.step()
                   │
                   ▼
          Update Parameters
"""


# ================================================================
#                NHÓM LỆNH CẦN NHỚ
# ================================================================

"""
TENSOR
------
torch.tensor()
torch.zeros()
torch.ones()
torch.rand()
torch.randn()
torch.arange()

SHAPE
-----
.shape
.ndim
.numel()
.reshape()
.view()
.flatten()
.unsqueeze()
.squeeze()

INDEXING
--------
x[0]
x[:, 0]
x[0, 1]

MATH
----
+
-
*
/
**
@
torch.matmul()

STATISTICS
----------
.sum()
.mean()
.max()
.min()
.argmax()
.argmin()

COMBINE
-------
torch.cat()
torch.stack()

NUMPY
-----
torch.from_numpy()
x.numpy()
x.cpu().numpy()

CUDA
----
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.device()
x.to(device)
x.cuda()
x.cpu()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.empty_cache()
torch.cuda.synchronize()

AUTOGRAD
--------
requires_grad=True
.backward()
.grad
.grad.zero_()
.detach()
torch.no_grad()

MODEL
-----
model.to(device)
model.train()
model.eval()

TRAINING
--------
optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()
"""