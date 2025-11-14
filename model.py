import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

def build_patternnet(input_dim, num_classes):
    """
    PatternNet = Lightweight shallow neural network (MLP style)
    Matches the paper's classifier design.
    """

    inputs = Input(shape=(input_dim,))

    x = Dense(256, activation="relu")(inputs)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model
