# import thư viện 
import numpy as np
import matplotlib.pyplot as plt

# 1. Định nghĩa hàm loss
# Giả sử mình có hàm loss 𝓛(θ) = θ⁴ - 6θ² + 4θ + 20
# khai báo hàm loss
def cal_loss_function(theta):
    return theta**4 - 6*theta**2 + 4*theta + 20

# Đạo hàm của hàm Loss 𝓛′(θ) = 4θ³ - 12θ + 4 (đạo hàm của θ⁴ - 6θ² + 4θ + 20))
def cal_loss_derivative(theta):
    return 4*theta**3 - 12*theta + 4

# 2. Vẽ đồ thị hàm loss
#Khai báo khoảng giá trị của theta
theta_values = np.linspace(-3, 3, 400)

#giá trị của hàm loss theo khoảng theta
loss_values = cal_loss_function(theta_values)

#Vẽ đồ thị hàm loss
plt.plot(theta_values, loss_values)

# 3. Khởi tạo
# Khởi tạo giá trị ban đầu của theta 
theta = 2.8
# Khởi tạo siêu tham số - hyperparameters
learning_rate = 0.01
# epsilon = 1e-4 
N = 100 # số vòng lặp

#Khai báo thêm
beta = 0.9
v = 0  # Vận tốc ban đầu

# cách 2: Khai báo số vòng lặp
for i in range(N):
    #tính giá trị hàm loss
    loss = cal_loss_function(theta)

    #tính giá trị đạo hàm của hàm loss theo theta
    grad = cal_loss_derivative(theta)

    # Vẽ điểm theta hiện tại 
    plt.plot(theta, loss, 'ro') # ro là điểm tròn màu đỏ

    #Dừng để quan sát
    plt.pause(0.1)

    
    # Cập nhật theta
    # theta = theta - learning_rate * grad
    
    # tính vận tốc
    v = beta*v + (1 - beta) * grad

    # cập nhật theta
    theta = theta - learning_rate*v

plt.show()
print("Giá trị theta tối ưu::::", theta)


