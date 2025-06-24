import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#  Phát sinh dữ liệu (giả lập thu thập từ thực tế)
def col_data():
    x = np.arange(2, 8, 0.2)
    noise = np.random.normal(0, 0.5, len(x))
    y = 3*x - 5 + noise  
    #Hàm ẩn: y = 3x - 5 + nhiễu (trong thực tế ta không biết các tham số này)
    return x , y

# Lấy dữ liệu
x, y = col_data()

# Đưa dữ liệu lên đồ thị để quan sát
plt.figure(figsize=(10, 6)) # Tăng kích thước biểu đồ
plt.plot(x, y, 'bo', label='Dữ liệu thực tế') # Vẽ điểm dữ liệu một lần
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dữ liệu')
plt.grid(True)

#khai báo tham số theta0, theta1
theta0 = 10
theta1 = -2

#khai báo các siêu tham số
alpha = 0.01
epsilon = 1e-4

i = 0
#lặp 
while True:
    der_theta0 = np.mean(theta0 + theta1*x - y)
    der_theta1 = np.mean((theta0 + theta1*x - y)*x)

    #np.mean là để lấy giá trị trung bình cộng

    if i % 50 == 0 or i < 50:
        x_vis = np.array([2, 8])
        y_vis = theta0 + theta1 * x_vis

        plt.plot(x_vis, y_vis)
        plt.pause(0.01)

    if np.abs(der_theta0) < epsilon and np.abs(der_theta1) < epsilon:
        break

    # cập nhật tham số theta
    theta0 = theta0 - alpha*der_theta0
    theta1 = theta1 - alpha*der_theta1

    i += 1


print("giá trị tối ưu của theta0:::", theta0)
print("giá trị tối ưu của theta1:::", theta1)
plt.show()