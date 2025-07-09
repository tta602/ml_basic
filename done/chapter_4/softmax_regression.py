# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

# Hàm vẽ ranh giới phân loại của mô hình Softmax
def plot_decision_boundary(X, y, theta):
    # Xác định phạm vi của lưới để vẽ biểu đồ
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    h = 0.01  # bước chia lưới

    # Tạo lưới điểm từ phạm vi X và Y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Ghép cặp điểm (x, y) thành vector 2D và thêm bias = 1
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_bias = np.hstack([grid, np.ones((grid.shape[0], 1))])

    # Tính xác suất dự đoán bằng hàm softmax
    probs = softmax(grid_bias @ theta)
    predictions = np.argmax(probs, axis=1)

    # Vẽ vùng quyết định và dữ liệu gốc
    plt.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    plt.title("Decision Boundary - Softmax Regression")
    plt.show()

# Hàm vẽ dữ liệu ban đầu với màu phân biệt các lớp
def plot_data(X, y, title="Raw Data"):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k', s=50)

    # Tạo chú thích (legend) cho từng lớp
    classes = np.unique(y)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(cls)), 
                          label=f"Class {cls}") for cls in classes]
    plt.legend(handles=handles)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

# Thiết lập seed để kết quả tái lập được
np.random.seed(100)

# Tạo dữ liệu giả lập: 100 mẫu, 2 đặc trưng, 3 lớp
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, n_classes=3
)

# Vẽ dữ liệu ban đầu
# plot_data(X, y)

#in vài giá trị y ban đầu
print(f"Giá trị y ban đầu, y[0] = {y[0]}, y[50] = {y[50]}, y[99] = {y[99]}")

#One-hot encoding cho nhãn y để huấn luyện với hàm softmax
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1, 1))

print(f"Giá trị Y khi đã encode, y[0] = {Y[0]}, y[50] = {Y[50]}, y[99] = {Y[99]}")

# hàm softmax 
def softmax(z):
    # Để ổn định số học bằng cách trừ cho max trước khi tính hàm mũ
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# hàm cross entropy
def cross_entropy(y_true, y_pred):
    # Tránh cho việc log(0)
    epsilon = 1e-15

    #giá trị của y_pred từ epsilon đến (1 - epsilon)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Hàm huấn luyện softmax bằng Gradient Descent
def train_softmax(X, y, lr=0.1, epochs = 1000):
    
    n_sample, n_feature = X.shape
    n_class = Y.shape[1]

    #thêm bias vào đặc trưng đầu vào
    X_bias = np.hstack([X, np.ones((n_sample, 1))])

    #khởi tạo tham số theta ngẫu nhiên
    theta = np.random.rand(n_feature + 1, n_class) * 0.01

    losses = []
    for epoch in range(epochs):
        z = X_bias @ theta

        y_pred = softmax(z)

        #tính giá trị hàm loss
        loss = cross_entropy(Y, y_pred)

        losses.append(loss)

        #Tính gradient của hàm mất mát theo theta
        grad = X_bias.T @ (y_pred - Y) / n_sample

        #Cập nhật tham số theta 
        theta = theta - lr*grad

        #hiển thị loss sau mỗi 100 epoch
        if epoch % 100 == 0:
            print(f"Epoch {epoch}::: Loss = {loss:.4f}")

        
    return theta, loss

#Huấn luyện mô hình
theta, losses = train_softmax(X, Y, lr=0.01, epochs=1000)

#vẽ ranh giới phân loại
plot_decision_boundary(X, y, theta)