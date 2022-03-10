"""
Training config for different datasets.
"""

dataset_config = {
    "cifar10": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.2,
        "ALPHA":0,
        "BETA":.1,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 11,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15, 
            "MAX_EPOCH": 6,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":1,
            "T_N_EPOCHS": 100,
            "TIME_STEPS": 11,
            "TEMPORAL_PERSISTENT": 1,
            "N_EPOCHS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":50,
        }
        
    },
    "online": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.2,
        "ALPHA":0,
        "BETA":1,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 11,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 6,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":1,
            "T_N_EPOCHS": 100,
            "TIME_STEPS": 10,
            "TEMPORAL_PERSISTENT": 1,
            "N_EPOCHS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":100,
        }
        
    },
    "mnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.4,
        "MAX_HAUSDORFF":0.25,
        "ALPHA":1,
        "BETA":1,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 11,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 6,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":1,
            "T_N_EPOCHS": 100,
            "TIME_STEPS": 10,
            "TEMPORAL_PERSISTENT": 1,
            "N_EPOCHS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":5,
        }
    },
    "mnist_full": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.4,
        "MAX_HAUSDORFF":0.25,
        "ALPHA":1,# 1
        "BETA":1,#1
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 20,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":5,
            "T_N_EPOCHS": 20,
            "TIME_STEPS": 50,
            "TEMPORAL_PERSISTENT": 1,
            "N_EPOCHS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":5,
        }
    },
    "fmnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":20.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.5,
        # "MAX_HAUSDORFF":.06, # for 10 epochs
        # "ALPHA":2,
        # "BETA":1.3,# larger, select less points from early epochs
        "MAX_HAUSDORFF":.06,
        "ALPHA":2,
        "BETA":1.3,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 10,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 10,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":1,
            "T_N_EPOCHS": 100,
            "N_EPOCHS": 5,
            "TIME_STEPS": 10,
            "TEMPORAL_PERSISTENT": 1,
            "N_EPOCHS": 5,   # the number of epochs to go through in one go
            "PATIENT": 4,    # early stopping patient
            "TEMPORAL_EDGE_WEIGHT":5,
        }
    },
    "fmnist_full": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":20.,
        "L_BOUND":0.5,
        # "MAX_HAUSDORFF":.06, # for 10 epochs
        # "ALPHA":2,
        # "BETA":1.3,# larger, select less points from early epochs
        "MAX_HAUSDORFF":.06,
        "ALPHA":2,
        "BETA":1.3,
        "INIT_NUM":300,
        "EPOCH_START": 1,
        "EPOCH_END": 50,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":3,
            "T_N_EPOCHS": 50,
            "PATIENT": 4,    # early stopping patient
        }
    },
    "cifar10_full": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "L_BOUND":0.6,
        "MAX_HAUSDORFF":0.2,
        "ALPHA":0,#0
        "BETA":.1,#.1
        "INIT_NUM":300,
        "EPOCH_START": 40,
        "EPOCH_END": 200,
        "EPOCH_PERIOD": 1,
        "training_config":{
            "N_NEIGHBORS":15,
            "MAX_EPOCH": 20,
            "S_N_EPOCHS": 5,
            "B_N_EPOCHS":1,
            "T_N_EPOCHS": 100,
            "PATIENT": 3,    # early stopping patient
        }
        
    },
}