import numpy as np

from local_model import local_ML_model
from data_slicing import create_list_of_sensors


def create_empty_array_list(blueprint_list):
    """
    list var = create_empty_array_list(blueprint_list)

    Returns new list containing same amount of arrays as blueprint_list.
    The arrays are of same shape as those in blueprint_list and are filled with zeros.

    :type  blueprint_list: list
    :param blueprint_list: an example of the list to be created. elements in blueprint_list need be of 'ndarray' type.
    :return: list with arrays of same shapes as the ones in the blueprint list.
    :rtype: list
    """
    empty_list = []
    for array in blueprint_list:
        empty_array = np.zeros(array.shape, dtype=np.float32)
        empty_list.append(empty_array)
    return empty_list


def get_total_number_of_samples(sensor_list):
    """
    int var = get_total_number_of_samples(sensor_list)

    Returns total number of samples of all sensor-type objects in sensor_list.

    :type sensor_list: list
    :param sensor_list: list containing Sensor-type objects
    :return: returns an integer depicting the total number of training samples of all Sensor-type objects combined.
    :rtype: int
    """
    number_of_samples = 0
    for sensor in sensor_list:
        number_of_samples += sensor.length_of_X_train()
    return number_of_samples


def make_scale_factor(sensor, total_number_of_samples):
    """
    int var = make_scale_factor(sensor, total_number_of_samples)

    Returns ratio between the number of training samples of one sensor and the total amount of samples of all sensors in list of sensors.
    :type sensor: sensor
    :param sensor: instance of class Sensor
    :type total_number_of_samples: int
    :param total_number_of_samples: The total number of training samples from sensors in list of sensors
    :return: sensors' training samples / total training samples
    :rtype: int
    """
    return sensor.length_of_X_train() / total_number_of_samples


def calculate_prediction_ratios(predictions_list):
    """
    int var =  calculate_prediction_ratios(predictions_list)

    Returns ratio between accurate predictions and inaccurate predictions.
    :type predictions_list: list
    :param predictions_list: list such of correct and incorrect predictions such as [correct, incorrect].
    :return: correct predictions / incorrect predictions.
    :rtype: int
    """
    correct_predictions = 0
    incorrect_predictions = 0
    for sensor_acc in predictions_list:
        correct_predictions += sensor_acc[0]
        incorrect_predictions += sensor_acc[1]
    ratio = correct_predictions / incorrect_predictions
    return ratio


def make_save_files(db_lines, round_ratios, test_description):
    """
    Creates two .txt files.\n
    <test_description>accs.txt contains list of all test accuracy values for all sensors.
    for example, if the number of sensors is 6 and the number of global rounds 50,
    then the .txt file will consist  300 lines.\n

    <test_description>rndratios.txt contains the values of the ratios between correct and incorrect predictions.
    if the number of rounds is 50 then the .txt file will consist  50 lines.\n
    :type db_lines: list
    :param db_lines: list of dictionaries containing key called 'test_accuracy'.
    :type round_ratios: list
    :param round_ratios: list of correct / incorrect predictions.
    :type test_description: str
    :param test_description: string describing the values of the hyperparameters chosen for the test.
    :return: None
    :rtype: None
    """
    list_of_accuracies = []
    for i in db_lines:
        list_of_accuracies.append(i['test_accuracy'])
    accs_filename = test_description + "accs.txt"
    ratios_filename = test_description + "rndratios.txt"
    np.savetxt(accs_filename, list_of_accuracies)
    np.savetxt(ratios_filename, round_ratios)


def global_model(sensor_data_list, number_of_rounds, epochs, num_of_lstm_layers, test_description):
    """
    global_model(sensor_data_list, number_of_rounds, epochs, num_of_lstm_layers, file_name)

    Runs a variant of the FedAvg algorithm (link to description in README).\n

    :type  sensor_data_list: list
    :param sensor_data_list: list containing K Sensor-class instances.
    :type  number_of_rounds: int
    :param number_of_rounds: number of global training rounds G.
    :type  epochs: int
    :param epochs: number of local training rounds E.
    :type  num_of_lstm_layers: int
    :param num_of_lstm_layers: number of hidden LSTM layers L the local model will use.  L in {1,2}
    :type  test_description: str
    :param test_description: string describing the test parameters. Used as name of .txt files

    :return: None
    :rtype: None

    See Also
        --------
        run_global_model()
        local_ML_model()
    """

    averaged_trainables = []
    dblines = []
    round_ratios = []

    # get total number of training samples in sensor_data_list
    total_number_of_samples = get_total_number_of_samples(sensor_data_list)

    # Global model starts running here
    for rnd in range(number_of_rounds):         # for each round rnd = 0,1,..., G
        print("Global round {}\n".format(rnd))

        predictions_one_round = []
        scaled_vars_one_round = []

        # collect and process results from local model.
        for sensor in sensor_data_list:  # for each sensor k = 0,1,..., K,

            scaled_vars = []
            scale_factor = make_scale_factor(sensor, total_number_of_samples)  # sf_k

            # run local model and collect accuracy data (see local_ML_model).
            trainable_vars, predictions, accuracies = local_ML_model(sensor,
                                                                     epochs,
                                                                     averaged_trainables,
                                                                     num_of_lstm_layers)

            dblines.append(accuracies)                  # save dbline data to dblines list
            predictions_one_round.append(predictions)   # save acc-data to predictions list

            # scale and save the weights and biases  w_i collected from the local_ML_model.
            for trainable_variable in trainable_vars:
                scaled_variable = scale_factor * trainable_variable
                scaled_vars.append(np.array(scaled_variable))

            scaled_vars_one_round.append(scaled_vars)

        # calculate ratio between correct and incorrect predictions and append them to list of ratios.
        ratios = calculate_prediction_ratios(predictions_one_round)
        round_ratios.append(ratios)

        #  aggregate scaled trainable variables.
        averaged_trainables = create_empty_array_list(blueprint_list=scaled_vars_one_round[0])  # placeholder for avg. variables

        for array_list in scaled_vars_one_round:  # for each list of scaled trainable variables
            for array_index in range(len(array_list)):          # for each index in the list
                averaged_trainables[array_index] += array_list[array_index]  #  put sum of scaled factors in each corresponding place.

        make_save_files(dblines, round_ratios, test_description)  # save accuracy and ratio data to .txt files.


def run_global_model(num_sensors,
                     num_bins,
                     global_rnds,
                     num_epochs,
                     num_lstm_layers,
                     primary_type,
                     secondary_type,
                     ):
    """
    Runs data collection functions and initializes global model.
    :type  num_sensors: int
    :param num_sensors: Number of virtual sensors
    :type  num_bins: int
    :param num_bins: Number of bins
    :type  global_rnds: int
    :param global_rnds: Number of global rounds
    :type  num_epochs: int
    :param num_epochs: Number of epochs
    :type  num_lstm_layers: int
    :param num_lstm_layers: Number of LSTM hidden layers in local model
    :type  primary_type: str
    :param primary_type: Primary data type
    :type  secondary_type: str
    :param secondary_type: Secondary data type
    :return: None
    :rtype: None
    """
    # gets raw data from server, prepares it and divides it in nr of sensors.
    list_of_sensors = create_list_of_sensors(primary_type,
                                             secondary_type,
                                             num_sensors,
                                             num_bins)

    # creates the start of the name of the save files.
    test_description = "dtype{}_{}rnds_{}ep_{}bins_".format(secondary_type, global_rnds, num_epochs,num_bins)

    # runs the global model. which itself will run the local model.
    global_model(list_of_sensors, global_rnds, num_epochs, num_lstm_layers, test_description)

