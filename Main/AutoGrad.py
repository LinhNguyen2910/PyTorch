import torch

# ================================================================
# 1. TẠO TENSOR CÓ THEO DÕI GRADIENT
x = torch.tensor(3.0, requires_grad=True)
# requires_grad=True:
# -> PyTorch sẽ theo dõi các phép tính liên quan đến x
# -> Sau này có thể tính đạo hàm của kết quả theo x


# ================================================================
# 2. TÍNH TOÁN VỚI TENSOR
y = x ** 2
# y = x^2
# Ta có:
# dy/dx = 2x


# ================================================================
# 3. BACKWARD - TÍNH ĐẠO HÀM
y.backward()
# backward() thực hiện quá trình Backpropagation
# PyTorch tự động tính gradient bằng Chain Rule


# ================================================================
# 4. LẤY GRADIENT
print(x.grad)
# x.grad = dy/dx
# Với x = 3:
# dy/dx = 2 * 3 = 6
# Kết quả: tensor(6.)


# ================================================================
# 5. VÍ DỤ COMPUTATION GRAPH
x = torch.tensor(3.0, requires_grad=True)
a = x ** 2       # a = x²
b = a + 2        # b = a + 2
y = b * 3        # y = 3b
y.backward()
print(x.grad)
# y = 3(x² + 2)
# dy/dx = 3 * 2x
#       = 6x
# x = 3 -> dy/dx = 18


# ================================================================
# 6. GRADIENT CỦA NHIỀU BIẾN
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = w * x + b
y.backward()
print(x.grad)   # ∂y/∂x
print(w.grad)   # ∂y/∂w
print(b.grad)   # ∂y/∂b


# ================================================================
# 7. GRADIENT CỦA NEURAL NETWORK ĐƠN GIẢN
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
y_true = torch.tensor(10.0)
y_pred = w * x + b
loss = (y_pred - y_true) ** 2
loss.backward()
print(w.grad)
print(b.grad)


# ================================================================
# 8. GRADIENT DESCENT THỦ CÔNG
learning_rate = 0.01
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad


# ================================================================
# 9. TẠI SAO DÙNG torch.no_grad()?
with torch.no_grad():
    w -= learning_rate * w.grad
# Khi cập nhật tham số:
# Không cần PyTorch xây dựng computation graph
# -> giúp tránh việc phép cập nhật cũng bị tính vào graph


# ================================================================
# 10. XÓA GRADIENT
w.grad.zero_()
# Gradient trong PyTorch mặc định được CỘNG DỒN
# Vì vậy trước mỗi lần backward mới:
# cần xóa gradient cũ


# ================================================================
# 11. detach()
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = y.detach()
# detach():
# Tạo tensor mới nhưng không còn nằm trong computation graph
# z vẫn có giá trị giống y
# nhưng không theo dõi gradient


# ================================================================
# 12. KIỂM TRA requires_grad
print(x.requires_grad)
# True -> tensor đang được theo dõi gradient


# ================================================================
# 13. KIỂM TRA GRADIENT
print(x.grad)
# Nếu chưa backward():
# thường sẽ là None
# Sau backward():
# x.grad chứa đạo hàm


# ================================================================
# 14. TẮT AUTOGRAD
with torch.no_grad():
    y = x ** 2
# Các phép tính bên trong no_grad()
# sẽ không được theo dõi gradient


# ================================================================
# 15. AUTOGRAD VỚI VECTOR / TENSOR
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
# y = [1, 4, 9]
# y là vector nên không thể gọi:
# y.backward()
# trực tiếp nếu y không phải scalar.
# Có thể lấy tổng thành scalar:
y.sum().backward()
print(x.grad)
# dy/dx = [2, 4, 6]


# ================================================================
# 16. ZERO GRADIENT
x.grad.zero_()
# Xóa gradient hiện tại:
# [2, 4, 6] -> [0, 0, 0]


# ================================================================
# 17. CÁCH VIẾT THƯỜNG GẶP TRONG TRAINING
for X, y in dataloader:
    # 1. Xóa gradient của batch trước
    optimizer.zero_grad()
    # 2. Forward propagation
    output = model(X)
    # 3. Tính Loss
    loss = criterion(output, y)
    # 4. Autograd tính gradient
    loss.backward()
    # 5. Optimizer cập nhật weight/bias
    optimizer.step()