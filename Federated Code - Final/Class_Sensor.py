from data_processing import init_to_db


class Sensor:
    """

    """
    instance_count = 0  # Used to enumerate instances

    def __init__(self,
                 primary_data_type,
                 secondary_data_type,
                 primary_data,
                 secondary_data,
                 number_of_bins,
                 number_of_steps=108
                 ):

        self.sensor_index = Sensor.instance_count
        self.name = "{}_{}".format("Sensor", self.sensor_index)
        self.primary_type = primary_data_type
        self.secondary_type = secondary_data_type
        self.LSTM_steps = number_of_steps
        self.number_of_bins = number_of_bins

        result_bins, df_in_train, num_of_bins, df_raw,\
        Xdf1_train, Xdf_TC_train, Xdf1_test, Xdf_TC_test,\
        Ydf_test, Ydf_train, df_X_pre, df_TC_pre, Xdf_train,\
        my_last_time\
            = init_to_db(primary_data, secondary_data, number_of_steps, number_of_bins)

        self.bins = result_bins
        self.num_of_bins = num_of_bins
        self.X_train = Xdf1_train
        self.TC_train = Xdf_TC_train
        self.Y_train = Ydf_train
        self.X_test = Xdf1_test
        self.TC_test = Xdf_TC_test
        self.Y_test = Ydf_test
        self.X_for_predict = df_X_pre
        self.TC_for_predict = df_TC_pre
        self.mytime = my_last_time
        self.output = "init"

        self.increment_instance_count()

    def __call__(self):
        self.LSTM_steps = 108
        result_bins, df_in_train, num_of_bins, df_raw, Xdf1_train, Xdf_TC_train, Xdf1_test, Xdf_TC_test, Ydf_test, Ydf_train, df_X_pre, df_TC_pre, Xdf_train, my_last_time = init_to_db(
            self.secondary_type, self.LSTM_steps)
        self.bins = result_bins
        self.num_of_bins = num_of_bins
        self.X_train = Xdf1_train
        self.TC_train = Xdf_TC_train
        self.Y_train = Ydf_train
        self.X_test = Xdf1_test
        self.TC_test = Xdf_TC_test
        self.Y_test = Ydf_test
        self.X_for_predict = df_X_pre
        self.TC_for_predict = df_TC_pre
        self.mytime = my_last_time
        self.output = "call"
        return self

    @staticmethod
    def increment_instance_count():
        Sensor.instance_count += 1

    def __repr__(self):
        length_of_datasets_for_training = "Length of X_train: {}.\n" \
                                          "Length of TC_train: {}.\n" \
                                          "Length of Y_train: {}.\n".format(self.length_of_X_train(),
                                                                            self.length_of_TC_train(),
                                                                            self.length_of_Y_train())
        length_of_datasets_for_testing = "Length of X_test: {}.\n" \
                                         "Length of TC_test: {}.\n" \
                                         "Length of Y_test: {}.\n".format(self.length_of_X_test(),
                                                                          self.length_of_TC_test(),
                                                                          self.length_of_Y_test())
        return "Length of datasets for training:\n"\
               + length_of_datasets_for_training + \
               "Length of datasets for testing:\n"\
               + length_of_datasets_for_testing

    def length_of_X_train(self):
        return len(self.X_train)

    def length_of_TC_train(self):
        return len(self.TC_train)

    def length_of_Y_train(self):
        return len(self.Y_train)

    def length_of_X_test(self):
        return len(self.X_test)

    def length_of_TC_test(self):
        return len(self.TC_test)

    def length_of_Y_test(self):
        return len(self.Y_test)

