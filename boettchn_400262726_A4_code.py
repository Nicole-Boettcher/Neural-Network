import numpy as np
import math
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
sc = StandardScaler()

random_seed = 7262 
num_nodes_layer_1 = 2
num_nodes_layer_2 = 2
num_epoch = 250

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
    X_train, X_test_val, t_train, t_test_val = train_test_split(feature_data, target_data, test_size = 0.2, random_state =random_seed)  # split data in 2 sets
    X_test, X_val, t_test, t_val = train_test_split(X_test_val, t_test_val, test_size = 0.5, random_state =random_seed)  # split data in 2 sets

    return np.array(sc.fit_transform(X_train)), np.array(t_train), np.array(sc.transform(X_val)), np.array(t_val), np.array(sc.transform(X_test)), np.array(t_test)

def init_weights(i):

    if i == 'Uniform 1':
        lower_bound_l1 = -1/math.sqrt(4)
        lower_bound_l2 = -1/math.sqrt(num_nodes_layer_1)
        lower_bound_l3 = -1/math.sqrt(num_nodes_layer_2)
        
        weights_layer_1 = np.random.uniform(lower_bound_l1, -1*lower_bound_l1, size=(num_nodes_layer_1, 5))
        weights_layer_2 = np.random.uniform(lower_bound_l2, -1*lower_bound_l2, size=(num_nodes_layer_2, num_nodes_layer_1+1))
        weights_output = np.random.uniform(lower_bound_l3, -1*lower_bound_l3, size=(1, num_nodes_layer_2+1))
        weights_layer_1[:, 0] = 0
        weights_layer_2[:, 0] = 0
        weights_output[:, 0] = 0

    elif i == 'Uniform 2':
        lower_bound_l1 = -math.sqrt(6/(4+num_nodes_layer_1))
        lower_bound_l2 = -math.sqrt(6/(num_nodes_layer_1+num_nodes_layer_2))
        lower_bound_l3 = -math.sqrt(6/(num_nodes_layer_2+1))
        
        weights_layer_1 = np.random.uniform(lower_bound_l1, -1*lower_bound_l1, size=(num_nodes_layer_1, 5))
        weights_layer_2 = np.random.uniform(lower_bound_l2, -1*lower_bound_l2, size=(num_nodes_layer_2, num_nodes_layer_1+1))
        weights_output = np.random.uniform(lower_bound_l3, -1*lower_bound_l3, size=(1, num_nodes_layer_2+1))
        weights_layer_1[:, 0] = 0
        weights_layer_2[:, 0] = 0
        weights_output[:, 0] = 0

    elif i == 'Kaiming He':
        weights_layer_1 = np.random.randn(num_nodes_layer_1, 5) * np.sqrt(2.0 / 4)
        weights_layer_2 = np.random.randn(num_nodes_layer_2, num_nodes_layer_1+1) * np.sqrt(2.0 / num_nodes_layer_1+1)
        weights_output = np.random.randn(1, num_nodes_layer_2+1) * np.sqrt(2.0 / num_nodes_layer_2 + 1)
        weights_layer_1[:, 0] = 0
        weights_layer_2[:, 0] = 0
        weights_output[:, 0] = 0
        
    return weights_layer_1, weights_layer_2, weights_output

def relu(x):
    return np.maximum(0, x)

def process_layer_test_1(inputs, weights, hidden_true):
    #print("Processing hidden layer")
    # output = h
    # z = weights * nodes
    # h = activate(z)

    # add dummy feature to inputs 
    inputs_w_dummy = np.insert(inputs, 0, 1, axis=1)

    z = np.dot(weights, inputs_w_dummy.T)     # should have size of (number of nodes, 1)
    if hidden_true:
        h = relu(z)     # should have size of (number of nodes, 1) -- check this works with array input to relu
    else:
        h = 1 / (1 + np.exp(-z))

    return z, h

def process_layer_test_2(inputs, weights, hidden_true):
    #print("Processing hidden layer")
    # output = h
    # z = weights * nodes
    # h = activate(z)

    # add dummy feature to inputs 
    inputs_w_dummy = np.insert(inputs, 0, 1, axis=0)

    z = np.dot(weights, inputs_w_dummy)     # should have size of (number of nodes, 1)
    if hidden_true:
        h = relu(z)     # should have size of (number of nodes, 1) -- check this works with array input to relu
    else:
        h = 1 / (1 + np.exp(-z))

    return z, h

def forward_propagation_test(x_train, weights_layer_1, weights_layer_2, weights_output):
    #print("Starting forward propagation")

    #init weights to all ones and segment by layers 

    # call process layer for each layer 
    # structure is 4 inputs, 1st hidden layer is 3 nodes, 2nd hidden layer is 3 nodes, output is 1 value with sigmoid
    
    z_1, output_layer_1 = process_layer_test_1(x_train, weights_layer_1, True)
    z_2, output_layer_2 = process_layer_test_2(output_layer_1, weights_layer_2, True)
    z_3, h_3 = process_layer_test_2(output_layer_2, weights_output, False) #z_3 is before sigmoid, h_3 is after sigmoid - h_3 = sigmoid(z_3)
    output_class = bayes_classifer(h_3)    #classify the probability
    return z_3, h_3, output_class

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

def forward_propagation(x_train, weights_layer_1, weights_layer_2, weights_output):
    #print("Starting forward propagation")

    #init weights to all ones and segment by layers 

    # call process layer for each layer 
    # structure is 4 inputs, 1st hidden layer is 3 nodes, 2nd hidden layer is 3 nodes, output is 1 value with sigmoid
    
    z_1, output_layer_1 = process_layer(x_train, weights_layer_1, True)
    z_2, output_layer_2 = process_layer(output_layer_1, weights_layer_2, True)
    z_3, final_output = process_layer(output_layer_2, weights_output, False)
    pred_class = bayes_classifer(final_output)    #classify the probability
    #print("z output = ", final_output)
    return z_1, output_layer_1, z_2, output_layer_2, z_3, final_output, pred_class

def bayes_classifer(output):
    return (output > 0.5).astype(int)
    
def back_propagation(x_train, t_train, weights_layer_1, z_1, output_layer_1, weights_layer_2, z_2, output_layer_2, weights_output, z_3):
    #print("Back Propagation")
    # Layer 3 calculations
    dell_J_z_3 = np.array(-int(t_train) + (1 / (1 + np.exp(-z_3))))    # equation (5) - ONE VALUE
    grad_W_3_J = dell_J_z_3 * (np.insert(output_layer_2, 0, 1))   # equation (7) - check h is horiztonal - tiz    

    W_3_trans = np.array(weights_output[:, 1:]).transpose()
    #W_3_trans = np.array([[weights_output[0][1]],[weights_output[0][2]],[weights_output[0][3]]])  #omit the first row -- WILL CHANGE WITH DIFF NUMBER OF NODE
    
    dell_z_2_J = np.multiply(derivative_relu(z_2).reshape(num_nodes_layer_2,1), dell_J_z_3*W_3_trans)  # (4) - WILL CHANGE WITH DIFF NUMBER OF ROWS
    # above this is correct 
    #print("Grad of layer 3 params (7) = ", grad_W_3_J)
    #print("equation 4 = ", dell_z_2_J)

    # Layer 2 calculations

    grad_W_2_J = dell_z_2_J * (np.insert(output_layer_1, 0, 1)) # equation (6) -- SHOULD THIS BE DOT PRODUCT?

    W_2_trans = np.array(weights_layer_2[:, 1:]).transpose()  #omit the first row -- WILL CHANGE WITH DIFF NUMBER OF NODE
    dell_z_1_J = np.multiply(derivative_relu(z_1).reshape(num_nodes_layer_1,1), W_2_trans.dot(dell_z_2_J)) # equation (3)
    #print("Grad of layer 2 params (6) = ", grad_W_2_J)
    #print("equation 3 = ", dell_z_1_J)

    # Layer 1

    grad_W_1_J = dell_z_1_J * (np.insert(x_train, 0, 1)).reshape(1,5)
    #print("Grad of layer 1 params = ", grad_W_1_J)

    return [grad_W_1_J, grad_W_2_J, grad_W_3_J]

def derivative_relu(z):
    conditional = z < 0
    return np.where(conditional, 0, 1)  # if each element in z is less than 0, than the slope is 0 and if z > 0 then the slope is 1

# def calculate_loss(x_set, t_set, weights_layer_1, weights_layer_2, weights_output):
#     # need to run all validation samples throught the net and get average cross entro loss
#     cross_entro_loss = 0
#     accuracy = 0
#     for index in range(len(x_set)):
#         pre_sigmoid, h_3, output_class = forward_propagation_test(x_set[index], weights_layer_1, weights_layer_2, weights_output)
#         cross_entro_loss += float(t_set[index])*np.logaddexp(0,-float(pre_sigmoid[0])) + (1-float(t_set[index]))*np.logaddexp(0,float(pre_sigmoid[0]))

#         if output_class == int(t_set[index]): 
#             accuracy = accuracy + 1

#     print("Accuracy = ", accuracy / len(t_set), "       cross_entro_loss = ", cross_entro_loss / len(x_set))
#     return cross_entro_loss / len(x_set)


# VECTORIZED loss function
def calculate_loss(x_set, t_set, weights_layer_1, weights_layer_2, weights_output):
    # need to run all validation samples through the net and get average cross entropy loss
    pre_sigmoid, _, _ = forward_propagation_test(x_set, weights_layer_1, weights_layer_2, weights_output)    
    # Compute the cross-entropy loss for all samples
    t_set = np.array(t_set).astype('float64')
    pre_sigmoid = np.array(pre_sigmoid).astype('float64')

    cross_entro_loss = np.average(t_set * np.logaddexp(0, -pre_sigmoid) + (1 - t_set) * np.logaddexp(0, pre_sigmoid))
    #print(cross_entro_loss)
    return cross_entro_loss # Negative sign to convert to average and handle minimization


def main():
    print("Assignment 4")
    #input_data = np.array((1372,5), float)

    x_train, t_train, x_test, t_test, x_val, t_val = pre_processing()
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    print(x_val.shape)
    print(t_val.shape)
    # need weights for each layer - 3 layers - [[3,5],[3,4],[1,4]]
    indices_train = np.arange(len(x_train))
    indices_val = np.arange(len(x_val))
    final_val_train_loss = []

    graph = 0
    alpha = 0.005
    weight_init = ['Uniform 1', 'Uniform 2', 'Kaiming He']
    for weights in weight_init:
        weights_layer_1, weights_layer_2, weights_output = init_weights(weights)
        training_cross_entro_loss = []
        validation_cross_entro_loss = []

        for epoch in range(num_epoch):
            #print("Epoch = ", epoch)
            np.random.shuffle(indices_train)
            x_train = x_train[indices_train]
            t_train = t_train[indices_train]
            np.random.shuffle(indices_val)
            x_val = x_val[indices_val]
            t_val = t_val[indices_val]

            for sample_num in range(len(x_train)):
                z_1, output_layer_1, z_2, output_layer_2, z_3, final_output, pred_class = forward_propagation(x_train[sample_num], weights_layer_1, weights_layer_2, weights_output)
                gradient_weights = back_propagation(x_train[sample_num], t_train[sample_num], weights_layer_1, z_1, output_layer_1, weights_layer_2, z_2, output_layer_2, weights_output, z_3)

                # update weights using gradient 
                weights_layer_1 = weights_layer_1 - alpha*(gradient_weights[0])
                weights_layer_2 = weights_layer_2 - alpha*(gradient_weights[1])
                weights_output = weights_output - alpha*(gradient_weights[2])

                # calculate training and validation error
                #print("Weights updated: ", sample_num)

            #print("Training: ")
            training_loss = calculate_loss(x_train, t_train, weights_layer_1, weights_layer_2, weights_output)
            #print("Validation: ")
            val_loss = calculate_loss(x_val, t_val, weights_layer_1, weights_layer_2, weights_output)
            #test_loss = calculate_loss(x_test, t_test, weights_layer_1, weights_layer_2, weights_output)
            #print("Val Loss = ", val_loss)
            #print("Training loss = ", training_loss)
            validation_cross_entro_loss.append(val_loss)
            training_cross_entro_loss.append(training_loss)
            #test.append(test_loss)


        graph += 1
        title = "Weight init: " , weights
        plt.subplot(3, 1, graph)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy loss")
        plt.ylim(0,0.2)
        plt.plot(training_cross_entro_loss, 'r:x', label="training")
        plt.plot(validation_cross_entro_loss, 'b:x', label="validation")
        #plt.plot(test, 'g:x', label="test")
        plt.legend()
        print(weights)
        print("Training loss = ", training_cross_entro_loss[-1])
        print("Validation loss = ", validation_cross_entro_loss[-1])
        print("Layer 1 weights: ", weights_layer_1)
        print("Layer 2 weights: ", weights_layer_2)
        print("Layer 3 weights: ", weights_output)
        final_val_train_loss.append(training_cross_entro_loss[-1])
        final_val_train_loss.append(validation_cross_entro_loss[-1])
        
    final = [round(element, 5) for element in final_val_train_loss]
    print(final)
    plt.suptitle("Training and Validation cross entropy loss per Epoch - (2,2)")
    plt.show()

    # correct_pred = 0
    # for test_sample_num in range(len(x_test)):
    #     z_3, pred_class = forward_propagation_test(x_test[test_sample_num], weights_layer_1, weights_layer_2, weights_output)
    #     #print("\ntarget = ", t_test[test_sample_num], " predicited = ", pred_class)
    #     if pred_class == int(t_test[test_sample_num]): 
    #         #print("correct")
    #         correct_pred = correct_pred + 1

    # print(weights_layer_1)
    # print(weights_layer_2)
    # print(weights_output)
    # print("accuracy = ", correct_pred / len(t_test))


    # correct_pred_train = 0
    # for train_sample_num in range(len(x_train)):
    #     pre_sigmoid, z_3, pred_class = forward_propagation_test(x_train[train_sample_num], weights_layer_1, weights_layer_2, weights_output)
    #     print("\ntarget = ", t_train[train_sample_num], " predicited = ", pred_class)
    #     if pred_class == int(t_train[train_sample_num]): 
    #         print("correct")
    #         correct_pred_train = correct_pred_train + 1

    # print("TRAINING accuracy = ", correct_pred_train / len(t_train))



main()

