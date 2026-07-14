SEED= 42
graph_type="Granger"

#Run Modes
DEBUG=True
USE_NETWORK=True
USE_MARKET_NETWORK=True
EXPLORATION_MODE=False

#Training and Pathing configurations
if DEBUG:
    TRAIN_START_DATE = "2017-01-01"
    TRAIN_END_DATE = "2018-01-01"
    TEST_START_DATE = "2018-01-02"
    TEST_END_DATE = "2019-06-01"
    CACHE_PATH = "processed_dow30_debug.pkl"
    RAW_CACHE_PATH = "raw_data_debug.pkl"
else:
    TRAIN_START_DATE = "2009-01-01"
    TRAIN_END_DATE = "2015-10-01"
    TEST_START_DATE = "2015-10-01"
    TEST_END_DATE = "2020-05-08"
    CACHE_PATH = "processed_dow30_full.pkl"
    RAW_CACHE_PATH = "raw_data_full.pkl"

#Network Configurations

NETWORK_THRESHOLD= 0.5
NETWORK_USE_ABS=True

#Indicator values to be used in the learning
INDICATORS = ['macd',
                'rsi_30',
                'cci_30',
                'dx_30']

if graph_type=="Correlation":
    NETWORK_INDICATORS=['degree','clustering','betweenness','strength']
    MARKET_LEVEL_NETWORK_INDICATORS = ['Average_Centrality', 'Average_Clustering','Average_Strength', 'Average_Betweenness']
elif graph_type =="Granger":
    NETWORK_INDICATORS=["granger_in_degree", "granger_out_degree", "granger_pagerank"]
    MARKET_LEVEL_NETWORK_INDICATORS = ["Average_granger_in_degree", "Average_granger_out_degree", "Average_granger_pagerank"]

else:
    print("ok")

#Training Parameters:


REBALANCE_WINDOW = 63 # rebalance_window is the number of days to retrain the model
VALIDATION_WINDOW = 63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)


# MODEL USED Arguments
A2C_model_kwargs = {
                    'n_steps': 5,
                    'ent_coef': 0.005,
                    'learning_rate': 0.0007
                    }

PPO_model_kwargs = {
                    "ent_coef":0.01,
                    "n_steps": 2048,
                    "learning_rate": 0.00025,
                    "batch_size": 128
                    }

DDPG_model_kwargs = {
                      #"action_noise":"ornstein_uhlenbeck",
                      "buffer_size": 10_000,
                      "learning_rate": 0.0005,
                      "batch_size": 64
                    }

TD3_model_kwargs = {
    "batch_size": 100,
    "buffer_size": 1_000_000,
    "learning_rate": 0.001
}

SAC_model_kwargs = {
    "batch_size": 100,
    "buffer_size": 1_000_000,
    "learning_rate": 0.001
}

if DEBUG:
    timesteps_dict = {
        "a2c": 1000,
        "ppo": 1000,
        "ddpg": 1000,
        "sac": 1000,
        "td3": 1000,
    }

else: 
    timesteps_dict = {'a2c' : 5000, 
                'ppo' : 5000, 
                'ddpg' : 5000,
                'sac': 5000,
                'td3': 5000
                }
    
#Run Name and Info:

from pathlib import Path
import json
from datetime import datetime

RUN_NAME = 'Test'

MAXLAG=5
