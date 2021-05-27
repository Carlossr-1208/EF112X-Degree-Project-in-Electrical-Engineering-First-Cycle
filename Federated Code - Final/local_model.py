
import math
import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers

# initialize variables for the first time.


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ----------------------------------------------------------------------------------------------------------

def local_ML_model(sensor_object,
                   number_of_epochs,
                   averaged_trainables,
                   number_of_LSTM_layers
                   ):
    """
    :type sensor_object: sensor
    :param sensor_object: an instance of a <class-Sensor> object.

    :type number_of_epochs: integer
    :param number_of_epochs: Number of local training rounds.

    :type averaged_trainables: list
    :param averaged_trainables: Averaged trainable variables.


    :param number_of_LSTM_layers: Either one or two.

    :return: Returns lists
    :rtype: list, list, list
    """

    batch_size = 30
    num_steps = sensor_object.LSTM_steps   #LSTM steps 108

    tf.reset_default_graph()               #resets tensorflow graph

    keep_prob = tf.placeholder("float")

    x_ = tf.placeholder(tf.float32, [None, num_steps, 1], name='input_placeholder')
    x_TC_ = tf.placeholder(tf.float32, [None, num_steps, 1], name='input_placeholder')

    # create rnn cells
    rnn_inputs = tf.unstack(x_, axis=1)
    rnn_inputs_TC = tf.unstack(x_TC_, axis=1)

    # 50 states per layer.
    if number_of_LSTM_layers == 1:
        num_of_states = [50]
    elif number_of_LSTM_layers == 2:
        num_of_states = [50, 50]
    else:
        print("Chose between 1 or 2 LSTM-layers.\n")

    regularizer = layers.l1_regularizer(0.0002)

    # create and running the LSTMs cells for measure type X and temperature
    with tf.variable_scope("PH", regularizer=regularizer):
        cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_of_states]  # LSTM layer
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        rnn_outputs, state = tf.nn.static_rnn(stacked_rnn_cell, rnn_inputs, dtype=tf.float32)
        output_st = rnn_outputs[-1]  # get the last state output
        output_reshape = tf.reshape(output_st, [-1, num_of_states[-1]])
        rnn_outputs_drop = tf.nn.dropout(output_reshape, keep_prob)

    with tf.variable_scope("TC", regularizer=regularizer):
        cells_TC = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_of_states]
        stacked_rnn_cell_TC = tf.nn.rnn_cell.MultiRNNCell(cells_TC)
        rnn_outputs_TC, state_TC = tf.nn.static_rnn(stacked_rnn_cell_TC, rnn_inputs_TC, dtype=tf.float32);
        output_st_TC = rnn_outputs_TC[-1]
        output_TC_reshape = tf.reshape(output_st_TC, [-1, num_of_states[-1]])
        rnn_TC_outputs_drop = tf.nn.dropout(output_TC_reshape, keep_prob)

    # Defining the dense layer's parameters
    W = weight_variable([num_of_states[-1], sensor_object.num_of_bins])  # Defining the dense layer's parameters
    b = bias_variable([sensor_object.num_of_bins])
    W_TC = weight_variable([num_of_states[-1], sensor_object.num_of_bins])

    ops_list = []

    if len(averaged_trainables) > 0: # Check if averaged variables were received.
        print("\t\t\tAveraged variables found. Creating ops_list\n")
        for i in range(len(averaged_trainables)):               # if received, assign the model's trainable variables with these values...
            trainable = tf.trainable_variables()[i].assign(averaged_trainables[i])
            ops_list.append(trainable)                          # ... and append these to ops_list to be initialized below.
    else:
        print("\t\t\tNo averaged variables found.Initializing from local model.\n")
        # If no were received the model will initialize with variables above.

    # combine the rnn results of m_type and TC
    y_predict = tf.nn.softmax(tf.matmul(rnn_outputs_drop, W) + tf.matmul(rnn_TC_outputs_drop, W_TC) + b)
    y_ = tf.placeholder("float", [None, sensor_object.num_of_bins])

    # Defining loss function for logistic regression model
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    reg_constant = 1.0
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
    loss = cross_entropy + reg_constant * reg_losses



    train_step = tf.train.AdamOptimizer(5e-4).minimize(loss)  # 5e-4 earning rate, Adam is an optimization algorithm based on SGD to update network weights
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))  # check matching, between output and predicting value, returns the truth value of (y_predict==y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # check the percentage that the difference between the prediction and the actual is less than 1 category
    close_prediction = tf.less_equal(tf.abs(tf.argmax(y_predict, 1) - tf.argmax(y_, 1)), 1)
    near_accuracy = tf.reduce_mean(tf.cast(close_prediction, "float"))

    init = tf.global_variables_initializer()

    db_line = {"sensor_{}".format(sensor_object.sensor_index): sensor_object.secondary_type,
               "timestamp_sensor": sensor_object.mytime}  # Preparing the string to insert prediction result into db

    # local_final_test_accuracy_list = []
    with tf.Session() as sess:
        data_train_size = sensor_object.X_train.shape[0]
        sess.run(init)
        # checks if there are averaged trainbles, and initilizes them if found
        if len(ops_list) > 0:
            for i in range(len(ops_list)):
                sess.run(ops_list[i])
        else:
            print("Initializing trainable variables.\n")

        for epoch in range(number_of_epochs + 1):

            number_of_batches = math.ceil(int(data_train_size / batch_size))    # batch rounded up
            list_of_random_numbers = np.random.permutation(number_of_batches)   # generate a list of random numbers
                                                                                # between zero and batch-1.
            for batch_number in range(number_of_batches):  # for every batch:

                train_index = list_of_random_numbers[batch_number] # get a random number from the index-list
                xx = sensor_object.X_train[train_index * batch_size:(train_index + 1) * batch_size]
                xx_TC = sensor_object.TC_train[train_index * batch_size:(train_index + 1) * batch_size]
                yy = sensor_object.Y_train[train_index * batch_size:(train_index + 1) * batch_size]

                train_step.run(session=sess, feed_dict={x_: xx, x_TC_: xx_TC, y_: yy,
                                                        keep_prob: 0.2})       #training step for variables

            # get the accuracy for training data
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={x_: sensor_object.X_train,
                                                      x_TC_: sensor_object.TC_train,
                                                      y_: sensor_object.Y_train, keep_prob: 1.0})
            train_near_accuracy = near_accuracy.eval(session=sess,
                                                     feed_dict={x_: sensor_object.X_train,
                                                                x_TC_: sensor_object.TC_train,
                                                                y_: sensor_object.Y_train, keep_prob: 1.0})

            print("\t\tEpoch: {}.\ttrain_accuracy = {}\ttrain_near_accuracy = {}".format(epoch, train_accuracy, train_near_accuracy))

            # get the accuracy for test data every 4th epoch
            if epoch % 4 == 0:
                train_accuracy = accuracy.eval(session=sess,
                                               feed_dict={x_: sensor_object.X_test,
                                                          x_TC_: sensor_object.TC_test,
                                                          y_: sensor_object.Y_test, keep_prob: 1.0})
                train_near_accuracy = near_accuracy.eval(session=sess,
                                                         feed_dict={x_: sensor_object.X_test,
                                                                    x_TC_: sensor_object.TC_test,
                                                                    y_: sensor_object.Y_test,
                                                                    keep_prob: 1.0})

                print("\t\tEpoch: {}.\ttest_accuracy = {}\ttest_near_accuracy = {}".format(epoch, train_accuracy,
                                                                                           train_near_accuracy))

        train_accuracy = accuracy.eval(session=sess,
                                       feed_dict={x_: sensor_object.X_test, x_TC_: sensor_object.TC_test,
                                                  y_: sensor_object.Y_test, keep_prob: 1.0})

        train_near_accuracy = near_accuracy.eval(session=sess,
                                                 feed_dict={x_: sensor_object.X_test,
                                                            x_TC_: sensor_object.TC_test,
                                                            y_: sensor_object.Y_test, keep_prob: 1.0})

        print("Final. tests_accuracy %g, tests_near_accuracy %g" % (train_accuracy, train_near_accuracy))

        # predict the output value
        prediction = y_predict.eval(session=sess,
                                    feed_dict={x_: sensor_object.X_for_predict,
                                               x_TC_: sensor_object.TC_for_predict,
                                               keep_prob: 1.0})

        correct_prediction_tensor = correct_prediction.eval(session=sess,
                                                            feed_dict={x_: sensor_object.X_test,
                                                                       x_TC_: sensor_object.TC_test,
                                                                       y_: sensor_object.Y_test, keep_prob: 1.0})


        accurate_prediction = 0
        inaccurate_prediction = 0
        # count number of correct and incorrect predictions.
        for bool_value in correct_prediction_tensor:
            if bool_value:  # == True:
                accurate_prediction += 1
            else:
                inaccurate_prediction += 1

        db_line['test_accuracy'] = train_accuracy.item()

        trainable_vars = tf.trainable_variables()
        trainvars = sess.run(trainable_vars)

        for i in range(sensor_object.num_of_bins):
            db_line['bin_' + str(i)] = [sensor_object.bins[i], sensor_object.bins[i + 1]]
            db_line['prob_' + str(i)] = prediction[0, i].item()

        return trainvars, [accurate_prediction, inaccurate_prediction], db_line
