import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from flask import Flask, render_template, request

# Loading the base models of reddit and twitter
reddit_rnn = tf.keras.models.load_model('models/base_models/reddit_rnn.h5')
reddit_lstm = tf.keras.models.load_model('models/base_models/reddit_lstm.h5')
reddit_gru = tf.keras.models.load_model('models/base_models/reddit_gru.h5')
reddit_bilstm = tf.keras.models.load_model('models/base_models/reddit_bilstm.h5')
reddit_bigru = tf.keras.models.load_model('models/base_models/reddit_bigru.h5')

twitter_rnn = tf.keras.models.load_model('models/base_models/twitter_rnn.h5')
twitter_lstm = tf.keras.models.load_model('models/base_models/twitter_lstm.h5')
twitter_gru = tf.keras.models.load_model('models/base_models/twitter_gru.h5')
twitter_bilstm = tf.keras.models.load_model('models/base_models/twitter_bilstm.h5')
twitter_bigru = tf.keras.models.load_model('models/base_models/twitter_bigru.h5')

# Loading the ensemble models
with open('models/ensemble_models/lr_meta_model_r.pkl', 'rb') as f:
    reddit_ensemble_model = pickle.load(f)

with open('models/ensemble_models/lr_meta_model_t.pkl', 'rb') as f:
    twitter_ensemble_model = pickle.load(f)

# Loading the meta-ensemble model
with open('models/meta_ensemble_model/final_meta_model.pkl', 'rb') as f:
    final_meta_ensemble_model = pickle.load(f)


app = Flask(__name__)
model = pickle.load(open("models/meta_ensemble_model/final_meta_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get the file from the request
        
        file = request.files["file"]
        
        df = pd.read_csv(file)  

        # Save the time column
        times = df['timestamp']

        # Defining the features for the model
        features = ['pos', 'neg', 'neu', 'close', 'volume']

        # Keeping the features
        df = df[features]

        # Shift the "close" column 1 hour into the future and make it the target variable
        df["target"] = df["close"].shift(-1)
        df = df.iloc[:-1]
        
        # Drop missing values
        df = df.dropna()

        # Split into features and target
        X = df.drop('target', axis=1).values
        y = df['target'].values.reshape(-1, 1)

        # Scale the data
        scaler_X = MinMaxScaler()
        X = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y = scaler_y.fit_transform(y)

        # Reshape input to be 3D [samples, timesteps, features]
        n_features = X.shape[1]
        X = X.reshape((X.shape[0], 1, n_features))

        # Generate predictions from the five models
        preds_test_rnn = twitter_rnn.predict(X)
        preds_test_lstm = twitter_lstm.predict(X)
        preds_test_gru = twitter_gru.predict(X)
        preds_test_bilstm = twitter_bilstm.predict(X)
        preds_test_bigru = twitter_bigru.predict(X)

        # Stack the predictions into a single matrix
        base_preds_test_twitter = np.column_stack((preds_test_rnn, preds_test_lstm, preds_test_gru, preds_test_bilstm, preds_test_bigru))

        # Generate predictions from the five models
        preds_test_rnn = reddit_rnn.predict(X)
        preds_test_lstm = reddit_lstm.predict(X)
        preds_test_gru = reddit_gru.predict(X)
        preds_test_bilstm = reddit_bilstm.predict(X)
        preds_test_bigru = reddit_bigru.predict(X)

        # Stack the predictions into a single matrix
        base_preds_test_reddit = np.column_stack((preds_test_rnn, preds_test_lstm, preds_test_gru, preds_test_bilstm, preds_test_bigru))

        reddit_test_preds = reddit_ensemble_model.predict(base_preds_test_reddit)
        twitter_test_preds = twitter_ensemble_model.predict(base_preds_test_twitter)

        # Combining both predictions as a numpy column
        meta_model_test_preds = np.column_stack((reddit_test_preds, twitter_test_preds))

        # Generating the final predictions
        final_prediction = final_meta_ensemble_model.predict(meta_model_test_preds)
        final_prediction = final_prediction.reshape(-1, 1) # reshape to match input shape
        final_prediction = scaler_y.inverse_transform(final_prediction)
        
        # Create a DataFrame with the actual and predicted values
        actual_values = scaler_y.inverse_transform(y)
        actual_values = actual_values.reshape(-1, 1)

        # Combine predicted and actual values into a single dataframe with time column
        predictions  = pd.DataFrame({'timestamp': times.iloc[-final_prediction.shape[0]:], 'actual': actual_values.flatten(), 'predicted': final_prediction.flatten()})

        return render_template('predictions.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
