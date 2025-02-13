from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model

def build_model(window_size: int):
    model = Sequential()
    model.add(InputLayer((window_size, 5)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(8, 'relu'))
    model.add(Dense(6, 'linear'))
    return model

def train_base_model(window_size: int, save_path: str, X_train, y_train, X_val, y_val, epochs=15, learning_rate=0.0001):
    model = build_model(window_size)
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )

    checkpoint = ModelCheckpoint(save_path, save_best_only=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[checkpoint]
    )

    return model


def train_tl_model(model_path: str, save_path: str, X_train, y_train, X_val, y_val, epochs=5, learning_rate=0.00001):
    tl_model = load_model(model_path)

    for layer in tl_model.layers[:-3]:
        layer.trainable = False

    tl_model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss=MeanSquaredError(), 
        metrics=[RootMeanSquaredError()]
    )

    checkpoint = ModelCheckpoint(save_path, save_best_only=True)

    tl_model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs,
        callbacks=[checkpoint]
        )
    
    return tl_model