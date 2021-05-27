from Class_Sensor import Sensor
from data_processing import get_raw_data_from_server
from data_processing import get_data_with_time_as_list


def down_sample(time_stamped_data, number_of_sensors):
    """
    list var = down_sample(time_stamped_data, number_of_sensors)

    Returns list of lists each containing down-sampled data.
    Divides [time_stamped_data] into [number_of_sensors] many lsits
    such that the first list l_1 gets data sample collected at time t.
    next list gets data collected at time t+1 and so on to the last list is created.
    Then the cycle repeats until all data in [time_stamped_data] has been distributed.

    :type  time_stamped_data: list
    :param time_stamped_data: data to be down-sampled.

    :type  number_of_sensors: int
    :param number_of_sensors: decides into how many lists the data will be divided.

    :return:    list of lists containing down-sampled data:
                the ith element in the list will contain the ith row of data and so on.
    :rtype: list

    Example:
        ______
        >>> ds_list = down_sample(['sample_1','sample_2','sample_3','sample_4'], 2)
        ds_list = [[sample_1, sample3], [sample_2, sample_4]]
    """
    return [time_stamped_data[i::number_of_sensors] for i in range(number_of_sensors)]


def create_list_of_sensors(primary_data_type, secondary_data_type, number_of_sensors, number_of_bins):
    """
    Returns list containing  [number_of_sensors] many <class 'sensor'> instances.
    :type primary_data_type: str
    :param primary_data_type: one of the following codes.
    'RX_C', 'PH_C', 'CN_A', 'SA_A','OS_D',
    :param secondary_data_type: The data type to be processed together with the primary one.
    :param number_of_sensors: Decides into how many parts the raw data will be divided in.
    :param number_of_bins:

    :return: A list of Sensor-type objects.
    """

    history_length_in_days = 365    # Collects data from the past year from database.

    # Collect, timestamp and down-sample raw primary data from server.
    raw_primary_data = get_raw_data_from_server(primary_data_type, history_length_in_days)
    raw_primary_data_timestamped = get_data_with_time_as_list(raw_primary_data, primary_data_type)
    down_sampled_primary_data = down_sample(raw_primary_data_timestamped, number_of_sensors)

    # Collect, timestamp and down-sample raw secondary data from server.
    raw_secondary_data = get_raw_data_from_server(secondary_data_type, history_length_in_days)
    raw_secondary_data_timestamped = get_data_with_time_as_list(raw_secondary_data, secondary_data_type)
    down_sampled_secondary_data = down_sample(raw_secondary_data_timestamped, number_of_sensors)
    list_containing_sensors = []

    for i in range(len(down_sampled_secondary_data)):
        # Creates as many Sensor-type objects as there are down-sampled data lists and distributes the data to them.
        sensor = Sensor(primary_data_type=primary_data_type,
                        secondary_data_type=secondary_data_type,
                        primary_data=down_sampled_primary_data[i],
                        secondary_data=down_sampled_secondary_data[i],
                        number_of_bins=number_of_bins)

        list_containing_sensors.append(sensor)
        sensor.sensor_index = i  # Used to give indexes to the sensors for identification.

    return list_containing_sensors
