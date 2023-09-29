import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Stonks - Sheet1-2.csv', header=0, index_col=0)
new_file = open('new_stonks.csv', 'w+')
new_file.close()
'''data = data.drop(["date"], axis=1)'''
data = data.drop(["NASDAQ"], axis=1)

df_max_scaled = data.copy()
resize = []
mins = []

for column in df_max_scaled.columns:
    resize.append(df_max_scaled[column].abs().max()-df_max_scaled[column].abs().min())
    mins.append(df_max_scaled[column].abs().min())
    df_max_scaled[column] = ((df_max_scaled[column]-df_max_scaled[column].abs().min()) /
                             (df_max_scaled[column].abs().max()-df_max_scaled[column].abs().min()))

i = 0
for column in df_max_scaled.columns:
    values = df_max_scaled[column].values
    train_X = []
    train_y = []
    n_future = 1
    n_past = 60

    for j in range(n_past+len(values)-350, len(values)-n_future+1):
        train_X.append(values[j-n_past:j])
        train_y.append(values[j+n_future-1:j+n_future])

    train_X, train_y = tf.convert_to_tensor(train_X), tf.convert_to_tensor(train_y)

    train_X, train_y = (tf.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1)),
                        tf.reshape(train_y, (train_y.shape[0], 1, 1)))
    print(train_X.shape, train_y.shape)

    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=15, batch_size=40,
                        validation_split=.2, verbose=1, shuffle=False)

    forecast = model.predict(train_X[-20:])
    forecast = forecast*resize[1] + mins[1]
    forecast = forecast.flatten()

    plt.plot([x for x in range(80)], df_max_scaled[column][-80:]*resize[1]+mins[1])
    plt.plot([x for x in range(80, 80+len(forecast))], forecast)
    plt.title(column)
    plt.show()

    i += 1