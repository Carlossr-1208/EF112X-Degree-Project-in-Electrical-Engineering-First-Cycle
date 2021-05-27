from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt


def get_raw_data_from_server(data_type, history_length_in_days):
    """
        Gets the raw data from server and returns it as a <cursor>-type (pymongo)
        my_measure_type = any one of: 'RX_C', 'PH_C', 'CN_A', 'SA_A','OS_D'
        history_length_in_days = # of days to grab data from
        function gets called by setting:
        variable = get_data_with_time_as_list(my_measure_type, history_length_in_days).

        :type my_measure_type: str
        :param my_measure_type: measure data type
        :type history_length_in_days: integer
        :param history_length_in_days: Number of days

        :return: list
    """

    server_details = "mongodb://user:password@129.192.69.145:27017/iWater"

    cluster = MongoClient(server_details)                                                               # connect to server
    my_data_base = cluster["iWater"]                                                                    #  switches to existing data base
    my_collection = my_data_base["iWater_node_01"]                                                      # creates collection named iWater_node_01

    time_sliding_window_start = get_sliding_window_start(history_length_in_days)                     # gets sliding time window.

    cursor_my_type = my_collection.find({'$and': [{'sensor': data_type},
                                                  {'timestamp_sensor': {'$gt': time_sliding_window_start}}]
                                         })

    return cursor_my_type


def get_data_with_time_as_list(data, data_type):
    """
    Returns list with time-stamped data and makes plots of the data used as input.
    """
    timestamped_data = pd.DataFrame(list(data))[['value', 'timestamp_sensor']]  # timestamps data

    # create .pdf images of the plots of input data
    name_dict = {'TC1_D': ["Water temperature ", "Temperature [℃]"],
                 'RX_C': ["Oxidation reduction potential", "Electric potential [mV]"],
                 'PH_C': ["pH levels over time", "pH"],
                 'CN_A': ["Electric conductivity", " Conductivity [\u03BCS/cm]"],
                 'SA_A': ["Salinity levels", "Salinity [ppt]"],
                 'OS_D': ["Oxygen saturation", "Dissolved oxygen [%]"]}


    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}

    matplotlib.rc('font', **font)
    plt.plot(timestamped_data['timestamp_sensor'], timestamped_data['value'])
    plt.legend()
    plt.ylabel(name_dict[data_type][1])
    plt.xlabel('Sample time [20 mins]')
    plt.title('{}'.format(name_dict[data_type][0]))
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.savefig(str("in_"+name_dict[data_type][0])+'.pdf')
    plt.show()

    return timestamped_data


def make_bins(sensor_data, number_of_bins):
    """ Puts data into 6 bins using qcut such that the numbers of the raw data in each bin are the same
        Returns output_bins, bins, df_my_type and my_last_time.
        The data that comes into bin_maker is the compartmentalized data from the pretend sensors.

        :type sensor_data: list
        :param sensor_data: sensor data
        :type number_of_bins: int
        :param number_of_bins: Number of bins

        :return: integer, integer, list, list

    """

    time_stamped_data = sensor_data

    df_my_type = time_stamped_data['value']
    df_time = time_stamped_data['timestamp_sensor'].tail(1)
    my_last_time = None

    for ddf_time in df_time:
        my_last_time = ddf_time
    # Cuts the data into equal sized bins, using percentiles based on the distribution of the data
    output_bins, bins = pd.qcut(df_my_type, number_of_bins, retbins=True, duplicates='drop')
    return output_bins, bins, df_my_type, my_last_time


def create_whole_time_series(data, num_steps, number_of_bins):
    """"
        Creates data with time series

        :type data: Sensor data list
        :param sensor data: sensor data
        :type num_steps: integer
        :param num_steps: Number of steps
        :type number_of_bins: integer
        :param number_of_bins: Number of bins

        :return: integer, list, list, list, list
        """

    # get variables from make_bins
    output_bins, \
    bins, \
    df_mytype, \
    my_last_time = make_bins(data, number_of_bins)

    df_new = []
    for i in df_mytype:
        j = first(range(1, len(bins)), lambda k: i <= bins[k]) - 1 # ??
        df_new.append(j)  # df_new is the whole time series of the quantized (from 0 to 5) where 5 is the bin number -1

    # To prepare the training data, such that each row of df_X is a time series of length num_steps=108, df_label is the label
    df_X, df_label = data_stacking(df_new, num_steps)   # To prepare the training data,
    df_X = np.reshape(df_X, (len(df_X), num_steps, 1))

    return bins, df_X, df_label, df_mytype, df_new, my_last_time  # ,df


def data_stacking(data_in, series_length):
    # preparing data for training,
    # from a series with length L to L-series_length number of series with length series_length
    X = []
    Y = []

    for i in range(0, len(data_in) - series_length, 1):
        sequence = data_in[i:i + series_length]
        label = data_in[i + series_length]
        X.append(sequence)
        Y.append(label)

    return X, Y


def first(the_iterable, condition=lambda x: True):  # fattar ej
    # getting the first element that satisfies the condition
    for i in the_iterable:
        if condition(i):
            return i


def get_sliding_window_start(history_length_in_days):
    """ Returns maximum amount of days to be checked in the data base.
        Used only in  'get_raw_data_from_server()'.

        :type history_length_in_days: integer
        :param history_length_in_days: Number of days

        :return: list
    """

    time_now = datetime.now()                               # gets date and time (real time), initial date & time
    time_delta = timedelta(days=history_length_in_days)     # gets the last 365 days, history_length = 365
    time_sliding_window_start = time_now - time_delta       # check at most d days int db, the start date & time
    # UPDATE_RONG
    time_change_date = datetime(2019, 12, 18);  # This is the date when the sensor was deployed in the lake
    time_sliding_window_start = max(time_sliding_window_start, time_change_date)
    # ENDUPDATE_RONG
    return time_sliding_window_start


def init_to_db(primary, secondary, num_steps, number_of_bins):
    """ Split the data in training data and test data, and then reshape it

              :type data: list
              :param data: measure data type
              :type tempdata: list
              :param tempdata: temperature data
              :type num_steps: integer
              :param num_steps: Number of steps
              :type number_of_bins: integer
              :param number_of_bins: Number of bins

              :return: integer, list, list, integer, list, list, list, list, list, list, list, list, list, list
    """
    # get variables from data_preparation
    result_bins, \
    df_in_train, \
    Y_onehot, \
    num_of_bins, \
    df_raw, \
    df_X_pre, \
    df_TC_pre, \
    my_last_time = data_preparation(primary, secondary, num_steps,number_of_bins)

    df_X_pre = np.reshape(df_X_pre, (1, len(df_X_pre), 1))  # normalization of the data for prediction, measure type X

    df_TC_pre = np.reshape(df_TC_pre, (1, len(df_TC_pre), 1))  # normalization of the data for prediction, temerature

    # shuffle the dataset and split for taining and validation, ratio 90/10
    Xdf_train, Xdf_test, Ydf_train, Ydf_test = train_test_split(df_in_train,
                                                                Y_onehot,
                                                                test_size=0.1,
                                                                shuffle=True)
    #skelearn
    Xdf1_train = np.reshape(Xdf_train[:, :, 0], (len(Xdf_train[:, :, 0]), num_steps, 1))

    Xdf_TC_train = np.reshape(Xdf_train[:, :, 1], (len(Xdf_train[:, :, 0]), num_steps, 1))

    Xdf1_test = np.reshape(Xdf_test[:, :, 0], (len(Xdf_test[:, :, 0]), num_steps, 1))

    Xdf_TC_test = np.reshape(Xdf_test[:, :, 1], (len(Xdf_test[:, :, 1]), num_steps, 1))

    return result_bins, \
           df_in_train, \
           num_of_bins, \
           df_raw, \
           Xdf1_train, \
           Xdf_TC_train, \
           Xdf1_test, \
           Xdf_TC_test, \
           Ydf_test, \
           Ydf_train, \
           df_X_pre, \
           df_TC_pre, \
           Xdf_train, \
           my_last_time


def data_preparation(primary_data, secondary_data, num_steps, number_of_bins):
    """read data from db, then prepare and organize the data
            normalize the data between 0 and 1, for prediction

            :type data: list
            :param data: measure data type
            :type tempdata: list
            :param tempdata: temperature data
            :type num_steps: integer
            :param num_steps: Number of steps
            :type number_of_bins: integer
            :param number_of_bins: Number of bins

            :return: integer, list, list, integer, list, list, list, list
        """

    # get the variables from create_whole_time_series for temperature
    TC_result_bins, \
    df_TC_train, \
    df_TC_label_train, \
    df_TC_raw, \
    df_TC_full, \
    _ = create_whole_time_series(primary_data, num_steps, number_of_bins)

    # get the variables from create_whole_time_series for measure type X
    result_bins, \
    df_X_train, \
    df_label_train, \
    df_raw, \
    df_full, \
    my_last_time = create_whole_time_series(secondary_data, num_steps, number_of_bins)

    num_of_bins = len(result_bins) - 1
    num_of_TC_bins = len(TC_result_bins) - 1
    df_X_train = df_X_train / float(num_of_bins)    # normalization, update the values of df_X
                                                    # from [0,binnumber-1] to [0,1].


    df_TC_train = df_TC_train / float(num_of_TC_bins)
    df_in_train = np.concatenate((df_X_train, df_TC_train), axis=2)  # combine X data and temperature data


    df_full2 = df_full.copy()

    for i in range(0, num_of_bins):     # This is to force the onehot process below has
                                        # the exact same number of classes as the bin number
        df_full2.append(i)

    Y_onehot = pd.get_dummies(df_full2)
    Y_onehot = Y_onehot[num_steps:-num_of_bins]

    df_pre = [xvalue / float(num_of_bins) for xvalue in df_full]  # normalization of the data for prediction
    df_TC_pre = [xvalue / float(num_of_bins) for xvalue in df_TC_full]  # df_TC_full/float(num_of_bins)

    return result_bins, df_in_train, Y_onehot, num_of_bins, df_raw, df_pre[-num_steps:], df_TC_pre[
                                                                                         -num_steps:], my_last_time
