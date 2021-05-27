
"""
Run code from here.
Chose how you want to test the code here.

Please observe: To be able to run the code one needs to input a username and password to access theDB Mongo server.
This is done in the function 'get_raw_data_from_server()' in the file 'data_processing.py'.
the variable holding this info is:
server_details = "mongodb://<user>:<password>@129.192.69.145:27017/iWater"
"""
from global_model import run_global_model



number_of_sensors = 6 # Number of subsets the raw data is divided to.
number_of_bins = 6  # Number of bins
global_rounds = 10   # Number of global rounds
num_of_epochs = 10   # Number of epochs
num_of_LSTM_layers = 1  # Number of LSTM hidden layers in local model
primary_data_type = 'TC1_D'  # Main data type
secondary_data_type = 'SA_A'  # Test data type
# ___________________________________________________________________________________________


run_global_model(number_of_sensors,
                 number_of_bins,
                 global_rounds,
                 num_of_epochs,
                 num_of_LSTM_layers,
                 primary_data_type,
                 secondary_data_type
                 )

