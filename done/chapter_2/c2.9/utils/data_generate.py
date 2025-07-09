import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class DataGenerator():
    def __init__(self, seed=42, mean=0, std=0.03):
        """
        Khởi tạo generator với seed cố định và tham số nhiễu Gaussian
        """
        self.seed = seed
        self.mean = mean
        self.std = std
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __generate_noise(self, size): 
        """
        Sinh nhiễu Gaussian với mean và std đã định nghĩa
        """
        return np.random.normal(self.mean, self.std, size)  

    def __generate_input(self, size):
        """
        Sinh đầu vào ngẫu nhiên phân bố đều trong [0, 1)
        """
        return np.random.rand(size)    

    def __target_function(self, x):
        """
        Hàm mục tiêu (ground truth) dùng để sinh y từ x
        """
        return np.sin(1 + x**2)

    def generate_true_data(self):
        """
        Sinh dữ liệu ground truth không nhiễu từ 100 điểm x đều nhau trong [0,1]
        """
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = self.__target_function(x).reshape(-1, 1)
        return x, y

    def generate_sample(self, size):
        """
        Sinh một tập dữ liệu gồm x, y (có nhiễu) và y_true (không nhiễu)
        """
        x = self.__generate_input(size=size)
        noise = self.__generate_noise(size=size)
        y_true = self.__target_function(x)
        y = y_true + noise
        return x.reshape(-1, 1), y.reshape(-1, 1)
    
    def generate_train_test(self, size):
        """
        Trả về tuple (train_data, test_data) với mỗi phần gồm (x, y, y_true)
        """
        train = self.generate_sample(size)
        test = self.generate_sample(size)
        return train, test


    def generate_named_dataset(self, name, size):
        """
        Sinh một tập dữ liệu được đặt tên, trả về dict dạng:
        {
            name: {
                'x_train': ...,
                'y_train': ...,
                'x_test': ...,
                'y_test': ...,
            }
        }
        """
        train, test = self.generate_train_test(size)
        x_train, y_train = train
        x_test, y_test = test

        return {
            name: {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
            }
        }
