import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

random_seed = 7262 
num_nodes_layer_1 = 3
num_nodes_layer_2 = 3

def pre_processing():
    feature_data = []
    target_data = []
    index = 0
    with open('data_banknote_authentication.txt') as f:
        for line in f.readlines():
            line_data = line.split(',')
            # split the target from the features 
            feature_data.append(line_data[0:4])
            target_data.append(line_data[4].replace("\n", ""))
            #print(feature_data[index])
            #print(target_data[index])
            index = index + 1

    # split the data
    X_train, X_test_val, t_train, t_test_val = train_test_split(feature_data, target_data, test_size = 0.3, random_state =random_seed)  # split data in 2 sets
    X_test, X_val, t_test, t_val = train_test_split(X_test_val, t_test_val, test_size = 0.5, random_state =random_seed)  # split data in 2 sets

    return np.array(sc.fit_transform(X_train)), np.array(t_train), np.array(sc.transform(X_val)), np.array(t_val), np.array(sc.transform(X_test)), np.array(t_test)

def relu(x):
    return np.maximum(0, x)


def process_layer(inputs, weights, hidden_true):
    #print("Processing hidden layer")
    # output = h
    # z = weights * nodes
    # h = activate(z)

    # add dummy feature to inputs 
    inputs_w_dummy = np.insert(inputs, 0, 1)

    z = weights.dot(inputs_w_dummy.transpose())     # should have size of (number of nodes, 1)
    if hidden_true:
        h = relu(z)     # should have size of (number of nodes, 1) -- check this works with array input to relu
    else:
        h = 1 / (1 + np.exp(-z))

    return z, h

def forward_propagation(x_train):
    print("Starting forward propagation")

    #init weights to all ones and segment by layers 

    weights_layer_1 = np.ones((num_nodes_layer_1, 5))
    weights_layer_2 = np.ones((num_nodes_layer_2, num_nodes_layer_1+1))
    weights_output = np.ones((1, num_nodes_layer_2+1))

    # call process layer for each layer 
    # structure is 4 inputs, 1st hidden layer is 3 nodes, 2nd hidden layer is 3 nodes, output is 1 value with sigmoid
    
    z_1, output_layer_1 = process_layer(x_train, weights_layer_1, True)
    z_2, output_layer_2 = process_layer(output_layer_1, weights_layer_2, True)
    z_3, final_output = process_layer(output_layer_2, weights_output, False)
    output_class = bayes_classifer(final_output)    #classify the probability
    print(output_class)
    return z_1, output_layer_1, z_2, output_layer_2, z_3, final_output

def bayes_classifer(output):
    if output <= 0.5:
        return 0
    else:
        return 1
    
def back_propagation(z_1, output_layer_1, z_2, output_layer_2, z_3, final_output, x_train):
    print("Back Propagation")
    dell_J_z_3 = -x_train + (1 / (1 + np.exp(-z_3)))    # equation (5)
    grad_W_3_J = dell_J_z_3 * (np.insert(output_layer_2, 0, 1))   # equation (7) - check h is horiztonal

    # need derivative of the activation function - relu derivative is 0 if input is <= 0 and 1 if input is > 0
    
    #dell_z_2_J = np.multiply(derivative_g(z_2),  )

#def derivative_g(z):
    

def main():
    print("Assignment 4")
    #input_data = np.array((1372,5), float)

    x_train, t_train, x_val, t_val, x_test, t_test = pre_processing()
    # need weights for each layer - 3 layers - [[3,5],[3,4],[1,4]]

    z_1, output_layer_1, z_2, output_layer_2, z_3, final_output = forward_propagation(x_train[0])
    back_propagation(z_1, output_layer_1, z_2, output_layer_2, z_3, final_output, x_train[0])

main()

