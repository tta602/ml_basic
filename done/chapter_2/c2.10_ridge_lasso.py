#import thư viện

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
#Ride là hồi quy tuyến tính điều chuần L2
#Lasso là hồi quy tuyến tính điều chuẩn L1

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#PolynomialFeatures tạo phi tuyến đặc trưng (feature)
#StandardScaler chuẩn hoá dữ liệu đầu vào

from sklearn.pipeline import make_pipeline
#tạo chuỗi xử lý

from sklearn.metrics import mean_squared_error
# tính giá trị hàm loss

from sklearn.model_selection import train_test_split
# chia tập dữ liệu 

# tạo dữ liệu phi tuyến
def generate_data(n_samples=100, seed=100):
    np.random.seed(seed)
    x = np.linspace(-2, 2, n_samples).reshape(-1, 1)
    y_true = 0.5 * x**2 - x + 2
    noise = np.random.normal(0, 1, size=(n_samples, 1))
    y = y_true + noise
    return x, y

# Tạo dữ liệu tổng thể
x_full, y_full = generate_data(n_samples=30)

# chia tập dữ liệu ra 80% train và 20% là test
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.6, random_state=42) 

#Khai báo các model

degree = 9
models = {
    "Linear Regression": LinearRegression(),
    f"Polynominal (deg={degree})": make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression()),
    f"Ridge (deg={degree})": make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), Ridge(alpha=1.0)),
    f"Lasso (deg={degree})": make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), Lasso(alpha=0.05, max_iter=10000))
}


plt.figure(figsize=(14, 9)) # Tăng kích thước biểu đồ

# Vẽ dữ liệu huấn luyện và kiểm tra
plt.scatter(x_train, y_train, color='blue', label="Dữ liệu huấn luyện", s=30, alpha=0.7)
plt.scatter(x_test, y_test, color='red', marker='x', label="Dữ liệu kiểm tra", s=50, alpha=0.7) # Dùng y_test

# Dải dữ liệu để vẽ đường dự đoán mượt mà (có thể dùng x_full hoặc một dải mới)
x_plot = np.linspace(x_full.min(), x_full.max(), 200).reshape(-1, 1)

# Vẽ hàm gốc nếu muốn so sánh
plt.plot(x_full, 0.5 * x_full**2 - x_full + 2, '--', color='black', label="Hàm gốc", linewidth=1.5)


colors = ['green', 'purple', 'orange', 'brown'] # Chọn màu khác để dễ phân biệt với điểm dữ liệu

for i, (name, model) in enumerate(models.items()):
    
    #huân luyện mô hình
    model.fit(x_train, y_train)

    #dự đoán và tính độ lỗi
    y_train_pred = model.predict(x_train)
    
    E_in = mean_squared_error(y_train, y_train_pred)

    #dự đoán và tính độ lỗi ở tập test
    y_test_pred = model.predict(x_test)

    E_out = mean_squared_error(y_test, y_test_pred)

    # Để vẽ đường dự đoán, sử dụng x_plot (dải giá trị rộng hơn)
    y_pred_plot = model.predict(x_plot)

    plt.plot(x_plot, y_pred_plot, color=colors[i], 
             label=f"{name} (E_in={E_in:.4f}, E_out={E_out:.4f})", linewidth=2)
    
    print(f"{name} (E_in={E_in:.4f}, E_out={E_out:.4f})")


plt.legend()
plt.title(f"So sánh Linear, Polynomial (deg={degree}), Ridge và Lasso")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()

