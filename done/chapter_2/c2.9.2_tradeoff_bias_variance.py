import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu mô phỏng cho biểu đồ Bias-Variance Tradeoff
# Model complexity (ví dụ: bậc đa thức)
complexity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

# Mô phỏng Bias (giảm dần khi phức tạp tăng)
# Bias thường được bình phương trong công thức tổng lỗi, nên có thể mô phỏng Bias^2 trực tiếp
bias_sq = np.array([10, 5, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]) * 1.5 

# Mô phỏng Variance (tăng dần khi phức tạp tăng)
variance = np.array([0.1, 0.3, 0.8, 2, 4, 7, 10, 13, 16, 20]) * 0.8

# Lỗi không thể giảm (Irreducible Error)
irreducible_error = 2

# Tổng lỗi
total_error = bias_sq + variance + irreducible_error

plt.figure(figsize=(10, 6))

plt.plot(complexity, bias_sq, label='Bias$^2$', color='blue', marker='o', linestyle='-')
plt.plot(complexity, variance, label='Variance', color='green', marker='o', linestyle='-')
plt.plot(complexity, total_error, label='Tổng Lỗi', color='red', marker='o', linewidth=3)

# Đánh dấu vùng Underfitting, Good Fit, Overfitting
plt.axvline(x=2.5, color='purple', linestyle='--', alpha=0.7, label='Vùng Underfitting')
plt.axvline(x=4.5, color='orange', linestyle='--', alpha=0.7, label='Vùng Overfitting')
# Có thể thêm một vùng tô màu nhẹ cho Good Fit
plt.axvspan(2.5, 4.5, color='gray', alpha=0.1, label='Vùng Optimal Complexity')


plt.xlabel('Độ phức tạp của mô hình (ví dụ: Bậc Đa thức)')
plt.ylabel('Giá trị lỗi')
plt.title('Sự đánh đổi Bias-Variance (Bias-Variance Tradeoff)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0) # Đảm bảo trục y bắt đầu từ 0

plt.tight_layout()
plt.show()