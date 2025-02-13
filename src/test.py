import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import load_model


def plot_predictions(model, X, y, start=0, end=100):
    time = [6, 12, 24, 48, 72, 168]
    predictions = model.predict(X)  
    num_forecast_steps = predictions.shape[1] 
    y = np.array(y)  

    fig, axes = plt.subplots(num_forecast_steps, 1, figsize=(8, 3 * num_forecast_steps))

    if num_forecast_steps == 1:
        axes = [axes]  

    for i in range(num_forecast_steps):
        df = pd.DataFrame({
            'Predictions': predictions[start:end, i],
            'Actuals': y[start:end, i]
        })
        
        axes[i].plot(df['Predictions'], label=f'Predictions ({time[i]}h)')
        axes[i].plot(df['Actuals'], label=f'Actuals ({time[i]}h)')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_title(f"Predictions vs Actuals for {time[i]}h")

    plt.tight_layout()
    plt.show()

    
    errors = [mse(y[:, i], predictions[:, i]) for i in range(num_forecast_steps)]
    
    return df, errors


def test_model(model_path: str, X_test, y_test):
    model = load_model(model_path)  
    df, errors = plot_predictions(model, X_test, y_test, start=0, end=100)  
    time = [6, 12, 24, 48, 72, 168]

    for i, error in enumerate(errors):
        print(f"Test MSE for {time[i]}h: {error:.4f}")

    return df, errors




