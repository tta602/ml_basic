import numpy as np
import time

# Sinh 2 vector ngẫu nhiên với 100 triệu phần tử (từ phân bố đều trong khoảng [0, 1))
a = np.random.rand(100_000_000)
b = np.random.rand(100_000_000)

# Dùng vector hóa
start = time.time()
dot_vector = np.dot(a, b)
end = time.time()
vector_time = end - start
print(f"Vectorization (np.dot): {vector_time:.6f} giây")

# Dùng vòng lặp
start_loop = time.time()
dot_loop = 0.0
for i in range(len(a)):
    dot_loop += a[i] * b[i]
end_loop = time.time()
loop_time = end_loop - start_loop
print(f"For-loop: {loop_time:.6f} giây")

# So sánh kết quả
print(f"Kết quả dot product (np.dot)::: {dot_vector:.4f}")
print(f"Kết quả dot product (for-loop): {dot_loop:.4f}")

# Thay vì duyệt từng phần tử bằng vòng lặp, np.dot() xử lý toàn bộ dữ liệu cùng lúc, nhanh và hiệu quả.
#Vector hóa giúp tính toán nhanh và hiệu quả hơn.
#Với dữ liệu lớn thì bắt buộc phải dùng vector hóa.