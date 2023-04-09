import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb

wandb.init(project='CICD_wandb', entity='manishrai727', name='keras-classification')


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions, model_name):
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.savefig(f'{model_name}.png', dpi=120)
    wandb.log({f'{model_name}.png': wandb.Image(plt)})


def mae(y_test, y_pred):
  return tf.metrics.mean_absolute_error(y_test, y_pred)
  

def mse(y_test, y_pred):
  return tf.metrics.mean_squared_error(y_test, y_pred)


X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)


N = 25
X_train = X[:N]
y_train = y[:N]

X_test = X[N:]
y_test = y[N:]

input_shape = X[0].shape 
output_shape = y[0].shape


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1), 
    tf.keras.layers.Dense(1)
    ])

model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ['mae'])


history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[wandb.keras.WandbCallback()])


y_preds = model.predict(X_test)
plot_predictions(train_data=X_train, train_labels=y_train,  test_data=X_test, test_labels=y_test,  predictions=y_preds)

mae_score = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
mse_score = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)
wandb.log({"Mean Absolute Error": mae_score, "Mean Squared Error": mse_score})
print(f'\nMean Absolute Error = {mae_score}, Mean Squared Error = {mse_score}.')

with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_score}, Mean Squared Error = {mse_score}.')

wandb.login()
