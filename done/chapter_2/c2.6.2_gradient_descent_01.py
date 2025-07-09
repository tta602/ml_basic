# import thư viện 
import numpy as np
import matplotlib.pyplot as plt

# 1. Định nghĩa hàm loss
# Giả sử mình có hàm loss 𝓛(θ) = θ² - 8θ + 10
# khai báo hàm loss
def cal_loss_function(theta):
    return theta**2 - 8*theta + 10

# Đạo hàm của hàm Loss 𝓛'(θ) = 2*θ - 8 (đạo hàm của θ² - 8θ + 10))
def cal_loss_derivative(theta):
    return 2*theta - 8

# 2. Vẽ đồ thị hàm loss
#Khai báo khoảng giá trị của theta
theta_values = np.linspace(-4, 12, 400)

#giá trị của hàm loss theo khoảng theta
loss_values = cal_loss_function(theta_values)

#Vẽ đồ thị hàm loss
plt.plot(theta_values, loss_values)

# 3. Khởi tạo
# Khởi tạo giá trị ban đầu của theta 
theta = 11 
# Khởi tạo siêu tham số - hyperparameters
learning_rate = 0.1
epsilon = 1e-4 

# cách 1: cài ngưỡng epsilon để dừng
while (True):
    #tính giá trị hàm loss
    loss = cal_loss_function(theta)

    #tính giá trị đạo hàm của hàm loss theo theta
    grad = cal_loss_derivative(theta)

    # kiểm tra giá trị của đạo hàm hàm loss có nhỏ hơn nhưỡng không?
    if np.abs(grad) < epsilon:
        break #Dùng và thoát khỏi vòng lặp

    # Vẽ điểm theta hiện tại 
    plt.plot(theta, loss, 'ro') # ro là điểm tròn màu đỏ

    #Dừng để quan sát
    plt.pause(0.1)

    # Cập nhật theta
    theta = theta - learning_rate * grad


plt.show()
# print("Giá trị theta tối ưu::::", theta)


