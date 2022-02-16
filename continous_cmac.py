import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
'''
35 weights
sample function at 100
training: 70, testing: 30
'''
def main():
    X = np.linspace(0, 2 * np.pi, 100)
    # X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    training_data_idx = np.random.choice(len(X), size=70, replace=False)
    training_data_idx.sort()
    training_data = [[X[i], Y[i]] for i in training_data_idx]
    testing_data_idx = []
    for i in range(100):
        if i not in training_data_idx:
            testing_data_idx.append(i)
    testing_data = [X[i] for i in testing_data_idx]
    weight = 0.0
    w_zero = []
    w_save =  w_zero
    error = mean_sq_array = total_time= gen_list = w_rand = weight_save = []
    start = time.time()     
    gen = 3
    mean_square_error = 1
    for g in tqdm(range(1, 37,2)):
        # generate the weight array with size 35 
        weight_arr = np.random.rand(35)
        padding = int((g - 1) / 2)
        # Creating padding on both ends
        weight_arr = np.append(np.zeros((padding)), weight_arr)
        weight_arr = np.append(weight_arr, np.zeros(padding))


        while mean_square_error > 0.01: 
            # Training 
            for j in range(0, 70):
                q = j / 2

                

                w_new_value =   weight / g
                y_train = training_data[j][1]

                e = y_train - w_new_value
                error = np.append(error, e)
                # Error Correction
                corrected_val = e / g  

                
                
                weight = 0


def main():
    X = np.linspace(0, 2 * np.pi, 100)
    # X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    training_data_idx = np.random.choice(len(X), size=70, replace=False)
    training_data_idx.sort()
    training_data = [[X[i], Y[i]] for i in training_data_idx]
    testing_data_idx = []
    for i in range(100):
        if i not in training_data_idx:
            testing_data_idx.append(i)
    testing_data = [X[i] for i in testing_data_idx]
    weight = 0
    error_arr = mean_sq_arr = total_time= gen_list = weight_arr_save = []
    start = time.time()   
    gen = 5
    mean_square_error = 1
    for g in tqdm(range(1, 37, 2)):
        # generate the weight array with size 35 
        weight_arr = np.random.rand(35)
        padding = int((g - 1) / 2)
        # Creating padding on both ends
        weight_arr = np.append(np.zeros((padding)), weight_arr)
        weight_arr = np.append(weight_arr, np.zeros(padding))

        while mean_square_error > 0.05: 
            # Training 
            for j in range(0, 70):
                q = int(j / 2)

                # calculate the sum of weight
                for k in range(1, g-1):
                    weight += weight_arr[k + q]
                weight += weight_arr[q] * 0.7
                weight += weight_arr[g-1 + q] * 0.3
                pred_output = weight / g
                y_train = training_data[j][1]

                error = y_train - pred_output
                error_arr = np.append(error_arr, error)
                # Error Correction
                corrected_val = error / g  

                for k in range(1, g-1):
                    weight_arr[k + q] += corrected_val
                weight_arr[q] += corrected_val * 0.7
                weight_arr[g - 1 + q] += corrected_val * 0.3

                weight = 0

            mean_square_error = np.mean(error**2)

        if g == gen:       
            weight_arr_save = weight_arr

        gen_list = np.append(gen_list, g)
        end = time.time()
        total_time = np.append(total_time, (end - start))
        mean_sq_arr = np.append(mean_sq_arr, mean_square_error)
        # reset mean_square_error 
        mean_square_error = 1

    gen_half = int(gen/2)
    # save weight_arr in every half of the gen size steps
    w_new = weight_arr_save[1::gen_half]
    w_new_array = []
    # Testing 
    for i in range(0, 30):
        q = int(i / 2)
        w_avg = np.sum(w_new[q - gen_half:q + gen_half + 1]) / gen
        w_new_array = np.append(w_new_array, w_avg)

    # plotting results
    plt.figure(1)
    plt.plot(X, Y, '-k', label="Input", linewidth=2)
    plt.plot(testing_data, w_new_array, '-r', label="Test")
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()