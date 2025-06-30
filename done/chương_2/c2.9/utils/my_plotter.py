import matplotlib.pyplot as plt
import numpy as np
import random

class MyPolynomialPlotter:
    def __init__(self, x_true, y_true):
        """
        x_true, y_true: ground truth để vẽ đường cong thực tế
        """
        self.x_true = x_true
        self.y_true = y_true

    def show_data_ground_truth(self, x_train, y_train, x_test, y_test):
        plt.figure(figsize=(4, 3))
        plt.scatter(x_train, y_train, color='blue', label='Tập huấn luyện (có nhiễu)', s=30)
        plt.scatter(x_test, y_test, color='red', marker='x', label='Tập kiểm thử (có nhiễu)', s=30)

        plt.plot(self.x_true, self.y_true, '--', color='black', label='Hàm gốc (ground truth)', linewidth=1.5)
        plt.legend()
        plt.grid(True)
        plt.title("Dữ liệu và Hàm mục tiêu")
        plt.xlabel("Đầu vào (x)")
        plt.ylabel("Đầu ra (y)")

        plt.show()

    def plot_polynomials(
        self, 
        name,
        x_train, 
        y_train, 
        x_test, 
        y_test, 
        predictions,
        color,
        scatter_data=True, 
        plot_predict=True, 
        degrees = []
    ):
        n = len(degrees)
        rows = (n + 2) // 3  # ví dụ: 7 degree thì cần 3 hàng
        fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for i, degree in enumerate(degrees):
            ax = axes[i]
            ax.plot(self.x_true, self.y_true, label="Ground Truth", color="black", linestyle="--", linewidth=2)

            y_train_pred, y_test_pred = predictions[degree]

            # Sắp xếp để vẽ đường mượt
            sorted_idx = np.argsort(x_train.flatten())
            x_sorted = x_train.flatten()[sorted_idx]
            y_pred_sorted = y_train_pred.flatten()[sorted_idx]

            if scatter_data:
                ax.scatter(x_train, y_train, color='blue', label=f"{name} Train", alpha=0.6, marker='o')
                ax.scatter(x_test, y_test, color='orange', marker='x', label=f"{name} Test", alpha=0.6)

            if plot_predict:
                ax.plot(x_sorted, y_pred_sorted, color=color, linewidth=2, label=f"{name} Train Fit")

            ax.set_title(f"Đa thức bậc {degree}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.legend()

        # Ẩn các subplot trống nếu degree không chia hết 3
        for k in range(i + 1, len(axes)):
            fig.delaxes(axes[k])

        plt.tight_layout()
        plt.show()
