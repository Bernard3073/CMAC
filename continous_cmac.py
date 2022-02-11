import matplotlib.pyplot as plt
import numpy as np
'''
35 weights
sample function at 100
training: 70, testing: 30
'''
def main():
    # X = np.linspace(0, 2 * np.pi, 100)
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    training_data_idx = np.random.choice(len(X), size=70, replace=False)
    training_data = [[X[i], Y[i]] for i in training_data_idx]
    testing_data_idx = []
    for i in range(100):
        if i not in training_data_idx:
            testing_data_idx.append(i)
    testing_data = [[X[i], Y[i]] for i in testing_data_idx]
    weight = 35
    weight_arr = np.zeros(weight).tolist()
    epoch_count = 1
    num_epoch = 100
    window_size = 10
    overlap = 35
    learning_rate = 0.1
    data_over_epoch = []
    overlap_data = []
    for o in range(1,overlap):
        while epoch_count <= num_epoch:

            pred_list = []
            error_list = []
            for i in range(70):
                cur = training_data[i][0]
                desire = training_data[i][1]
                
                associate_func = weight * (cur/window_size)
                lower = 0 if associate_func - (o/2) < 0 else associate_func - (o/2) 
                upper = weight-1 if associate_func + (o/2) > weight-1 else associate_func + (o/2) 
                # print(lower, upper)
                pred = 0
                for j in range(int(lower)+1, int(upper)):
                    pred += weight_arr[j] * cur
                pred += weight_arr[int(lower)] * 0.8
                pred += weight_arr[int(upper)] * 0.2
                pred_list.append(pred)

                error = desire - pred
                error_list.append(error)
                # error correction
                corrected_val = error / overlap
                # update weights
                for i in range(int(lower)+1, int(upper)):
                    weight_arr[i] += learning_rate * corrected_val
                weight_arr[int(lower)] += learning_rate * corrected_val * 0.8
                weight_arr[int(upper)] += learning_rate * corrected_val * 0.2
            
            data_over_epoch.append([pred, error, weight_arr])
            epoch_count += 1
        # testing
        error_list_testing = []
        pred_list_testing = []
        for i in range(30):
            cur = testing_data[i][0]
            desire = testing_data[i][1]
                
            associate_func = weight * (cur/window_size)
            lower = 0 if associate_func - (o/2) < 0 else associate_func - (o/2) 
            upper = weight-1 if associate_func + (o/2) > weight-1 else associate_func + (o/2) 
            # predict output
            pred = 0
            for j in range(int(lower)+1, int(upper)):
                pred += weight_arr[j] * cur
            pred += weight_arr[int(lower)] * 0.8
            pred += weight_arr[int(upper)] * 0.2
            pred_list_testing.append(pred)
            # calculate error
            error = desire - pred
            error_list_testing.append(error)

        plt.plot(X, Y)
        plt.plot([testing_data[i][0] for i in range(30)], pred_list_testing)
    plt.show()
if __name__ == '__main__':
    main()