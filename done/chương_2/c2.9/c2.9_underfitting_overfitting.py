import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from utils.data_generate import DataGenerator
from utils.my_polynominal_regression import MyPolynomialRegressionGD
from utils.my_plotter import MyPolynomialPlotter

def show_results_table(results):
    """
    Hiển thị bảng kết quả E_in và E_out theo từng bậc (degree)
    """
    print(f"{'Degree':<10}{'E_in':<15}{'E_out':<15}")
    print("-" * 40)
    for degree, E_in, E_out in results:
        print(f"{degree:<10}{E_in:<15.6f}{E_out:<15.6f}")

seed = 100
data_gen = DataGenerator(seed=seed, mean=0, std=0.03)

x_true, y_true = data_gen.generate_true_data()

size = 100
x_train, y_train = data_gen.generate_sample(size=size)
x_test, y_test = data_gen.generate_sample(size=size)

plotter = MyPolynomialPlotter(x_true, y_true)
plotter.show_data_ground_truth(x_train, y_train, x_test, y_test)
degree_list= [1, 3, 9]
model = MyPolynomialRegressionGD()
results, predictions = model.train_and_evaluate(x_train, y_train, x_test, y_test, degree_list)
show_results_table(results)

plotter.plot_polynomials(
    'D1',
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    predictions,
    color = 'blue',
    scatter_data=True, 
    plot_predict=True, 
    degrees = degree_list
)

plt.show()






