import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Hàm tạo dữ liệu phân loại nhị phân
def generate_binary_classification_data(n_samples=200, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)

    # Gán nhãn dựa trên tổng hai đặc trưng
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Dịch chuyển điểm để tạo khoảng cách giữa hai lớp
    X[y == 0] -= 1.8
    X[y == 1] += 1.8

    # Thêm nhiễu nhẹ
    X += np.random.normal(0, 0.5, X.shape)
    return X, y

# Vẽ dữ liệu 
def plot_data(X, y):
    plt.figure(figsize=(9, 7))

    # Vẽ các điểm dữ liệu
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', 
                label='Lớp 0', s=80, edgecolors='w')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', 
                label='Lớp 1', s=80, edgecolors='w')

    plt.title("Logistic Regression - Phân loại nhị phân")
    plt.xlabel("Đặc trưng 1 (X1)")
    plt.ylabel("Đặc trưng 2 (X2)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_decision_boundary(X, y, theta, resolution=0.02):
    plt.figure(figsize=(9, 7))

    # --- 1. Tạo lưới điểm trong không gian ---
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # --- 2. Dự đoán xác suất cho từng điểm trong lưới ---
    grid_points = np.c_[np.ones((xx.ravel().shape[0], 1)), 
                        xx.ravel(), yy.ravel()]
    probs = sigmoid(grid_points @ theta).reshape(xx.shape)

    # --- 3. Vẽ nền màu dựa trên xác suất ---
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], 
                 alpha=0.2, colors=['blue', 'red'])

    # --- 4. Vẽ dữ liệu ---
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', 
                label='Lớp 0', s=80, edgecolors='w')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', 
                label='Lớp 1', s=80, edgecolors='w')

    # --- 5. Vẽ đường phân chia (decision boundary) ---
    x_vals = np.array([x_min, x_max])
    y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]
    plt.plot(x_vals, y_vals, color='black', 
             linewidth=2.5, label='Đường phân chia')

    plt.title("Logistic Regression - Vùng quyết định và đường phân chia")
    plt.xlabel("Đặc trưng 1 (X1)")
    plt.ylabel("Đặc trưng 2 (X2)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

# --- Chạy chương trình ---
X, y = generate_binary_classification_data(n_samples=200)
plot_data(X, y)

#khai báo các siêu tham số
# learning_rate = 0.1
# iterations = 1000

# # # khai báo theta là random
# theta = np.random.randn(3)

# X_bias = np.c_[np.ones((X.shape[0], 1)), X] 

# # Lặp
# for i in range(iterations):
    
#     z = X_bias @ theta

#     y_pred = sigmoid(z)

#     gradient = (X_bias.T @ (y_pred - y)) / len(y)

#     #cập nhật trọng số theta
#     theta = theta - learning_rate * gradient

# khai báo model
model = LogisticRegression(solver="liblinear")

#huấn luyện model
model.fit(X, y)

theta = np.concatenate(([model.intercept_[0]], model.coef_[0]))
#[bias, theta1, theta2]

print("Giá trị theta tối ưu:::", theta)

plot_decision_boundary(X, y, theta)


