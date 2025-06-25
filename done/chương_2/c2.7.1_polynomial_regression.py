import numpy as np
import matplotlib.pyplot as plt

# --- 1. Tạo dữ liệu giả lập phi tuyến tính ---
def generate_polynomial_data(num_samples=100):
    """
    Tạo dữ liệu giả lập có dạng phân bố phi tuyến tính (ví dụ: parabol).
    """
    np.random.seed(42) # Để kết quả có thể lặp lại

    # x từ -5 đến 5, dạng cột vector
    x = np.linspace(-5, 5, num_samples).reshape(-1, 1) 
    
    # Giả sử hàm thực tế là y = 0.5*x^2 - 2*x + 3  -> hàm đa thức bậc 2
    y_true = 0.5 * x**2 - 2 * x + 3 

    noise = np.random.normal(0, 0.5, num_samples).reshape(-1, 1)

    y = y_true + noise # Thêm nhiễu vào dữ liệu
    
    # Trả về x, y có nhiễu
    return x, y

m = 100 # sỗ mẫu dữ liệu
x_raw, y_data = generate_polynomial_data(m)

# Hiển thị dữ liệu ban đầu
plt.figure(figsize=(10, 6))
plt.plot(x_raw, y_data, 'o', label='Dữ liệu thực tế', alpha=0.7, markersize=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dữ liệu ban đầu có dạng phi tuyến tính')
plt.grid(True)

plt.show()

#khai báo theta
theta = np.random.randn(3,1)

alpha = 0.01
num_interatons = 10000

#khai báo X
one_col = np.ones((m, 1))

X = np.c_[one_col, x_raw, x_raw**2]

Y = y_data.reshape(m, 1)

loss_history = [] # để lưu lại giá trị hàm loss sau mỗi vòng lặp

# lặp
for i in range(num_interatons):
    
    # tính Xθ 
    Y_hat = np.dot(X, theta)

    #tính sai số
    error = Y_hat - Y

    # tính giá trị hàm loss
    loss = (1/(2*m))*np.sum(error**2)
    loss_history.append(loss)

    # tính napla
    napla = (1/m) * np.dot(X.T, error)

    #cập nhật tham số theta
    theta = theta - alpha*napla


print("Giá trị theta tối ưu:::", theta)


plt.figure(figsize=(10,7))

#vẽ dữ liệu thực tế
plt.plot(x_raw, y_data, 'o', label ="Dữ liệu thực tế", alpha = 0.7, markersize = 5)

# tính giá trị Y dự đoán dựa vào theta tối ưu
Y_pred_plot = np.dot(X, theta)

#Vẽ đường cong đã học
plt.plot(x_raw, Y_pred_plot, '-', color='red', linewidth=3, 
         label="Đường hồi quy đa thức bậc 2")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mô hình hồi quy đa thức bậc 2 với Gradient Descent')

#hiển thị phần chú thích
plt.legend()
plt.grid(True)

plt.show()

# vẽ đồ thị lịch sử hàm loss
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Lịch sử hàm loss")
plt.xlabel('Số vòng lặp')
plt.ylabel('Giá trị hàm loss (MSE)')
plt.grid(True)
plt.show()
