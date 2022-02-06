import matplotlib.pyplot as plt
import numpy as np
'''
35 weights
sample function at 100
training: 70, testing: 30
'''
def main():
    X = np.linspace(0, 2 * np.pi, 100)
    Y = np.cos(X)
    training_data_idx = np.random.choice(len(X), size=70, replace=False)
    training_data = [[X[i], Y[i]] for i in training_data_idx]
    testing_data_idx = []
    for i in range(100):
        if i not in training_data_idx:
            testing_data_idx.append(i)
    testing_data = [[X[i], Y[i]] for i in testing_data_idx]
    hash_func = dict()
    weight = 35
    epoch_count = 1
    num_epoch = 100
    window_size = 5
    while epoch_count <= num_epoch:
        error_sum = 0
        for i in training_data:
            cur = training_data[i][0]
            desire = training_data[i][1]
            associate_func = weight * (cur/window_size)
	
if __name__ == '__main__':
    main()