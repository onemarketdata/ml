from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow import random

def dnn(meta, compile_kwargs, **kwargs):
    random.set_seed(kwargs.get('random_seed', 42))
    model = Sequential()
    model.add(Input(shape=(meta['n_features_in_'])))
    for layer in range(1, kwargs.get('hid_layers_num', 2) + 1):
        model.add(Dense(kwargs.get(f'neurons_num_layer{layer}', 4),
                        activation=kwargs.get(f'activation_layer{layer}', 'relu')))
        if kwargs.get(f'dropout_layer{layer}', 0) > 0:
            model.add(Dropout(kwargs[f'dropout_layer{layer}']))
    model.add(Dense(1))
    model.compile(loss=compile_kwargs["loss"],
                optimizer=compile_kwargs["optimizer"])
    return model