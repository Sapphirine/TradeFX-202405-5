import numpy as np
import pandas as pd
import pickle
import warnings
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from .ForexEnviroment import ForexEnvironment
warnings.filterwarnings("ignore")


def predict(currencyPair):
    # Load the saved model
    with open('./models/'+currencyPair+'.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    X = pd.read_csv('./data/'+currencyPair+'_H1.csv')[['time', 'close']]
    X['time'] = pd.to_datetime(X['time'])
    X = X.set_index('time')
    #print(X.head())

    # Define the start date
    start_date = "2022-02-22 10:00:00"

    # Generate a DatetimeIndex for each hour for a year after the start date
    date_range = pd.date_range(start=start_date, periods=365*24, freq='H')

    # Create a DataFrame with the generated index
    X_test = pd.DataFrame(index=date_range)

    # Add a 'close' column with placeholder values
    X_test['close'] = 0.0
    # X_test = X_test.set_index('time')
    # X_test = X_test[X_test.index >= '2020-05-01']
    # X_test = X_test[X_test.index <= '2021-05-02']

    # Reset the index of X_test to obtain integer indices
    X_test_reset = X_test.reset_index()

    # Use the predict method to generate forecasts
    predictions_array = loaded_model.predict(start=0, end=len(X_test_reset)-1)

    # Create a DataFrame from the predictions array
    predictions_df = pd.DataFrame(predictions_array, index=X_test.index, columns=[
                                  'Predictions']).iloc[1:]

    predictions_df.to_csv('./data/predictions_'+currencyPair+'.csv')

    X_index_array = X.index.to_numpy()
    X_price = X['close'].to_numpy()

    predict_index_array = predictions_df.index.to_numpy()[1:]
    predict_price_array = predictions_df['Predictions'].to_numpy()[1:]


    # Concatenate index arrays
    combined_index_array = np.concatenate((X_index_array, predict_index_array))

    # Concatenate price arrays
    combined_price_array = np.concatenate((X_price, predict_price_array))


    combined_df = pd.DataFrame(data=combined_price_array, index=combined_index_array, columns=['Predictions'])

    combined_df.to_csv('./data/predictions_combined_'+currencyPair+'.csv')


    return predictions_df


def init(currency_pair, initial_value, window_size, data):
    env = ForexEnvironment(
        observations=data, initial_value=initial_value, window_size=window_size)
    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose=1, n_steps=128, gamma=0.99, ent_coef=0.01, learning_rate=0.0005,
                tensorboard_log="./tensorboard/ForexEnvironment_"+currency_pair)
    return model


def train(model, window_size):
    for i in range(1):
        model.learn(total_timesteps=window_size, reset_num_timesteps=False)
    return model


def evl(initial_value, window_size, data, frame_length, model):

    # # Create the ForexEnvironment instance
    testenv = ForexEnvironment(
        initial_value, observations=data, window_size=window_size)

    temp_filename = testenv.evaluate(
        frame_length, model=model, marker_size=25, verbose=1)

    return temp_filename
