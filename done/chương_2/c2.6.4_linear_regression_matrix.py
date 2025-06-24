import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def col_data():
    x = np.arange(2, 8, 0.2)
    noise = np.random.normal(0, 0.5, len(x))
    y = 3*x - 5 + noise
    return x , y

x, y = col_data()


# khởi tạo tham số
theta = np.array([[10],[-2]])
alpha = 0.01 #learning rate
epsilon = 1e-4


m = len(x)

# Bước 1: Tạo một mảng gồm toàn số 1 có cùng số hàng với x
one_column = np.ones((m, 1))

#Ghép cột số 1 vào phía trước mảng x ban đầu
X = np.c_[one_column, x]

Y = y.reshape(m, 1)

#Lặp
while True:
    napla = (1/m)*np.dot(X.T, (np.dot(X, theta) - Y))
   
    # Kiểm tra điều kiện dừng trước khi cập nhật theta
    if np.all(np.abs(napla) < epsilon):
        break
    
    theta = theta - alpha*napla

    
print("Giá trị tối ưu theta0::::", theta)
plt.show()