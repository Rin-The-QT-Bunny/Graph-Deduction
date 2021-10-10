import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import setup

# Load the setting of the model
args = setup.load_setup()
dim_word = args["dim_words"]
dim_entity = args["dim_entity_id"]

# Build the soft neural operators
relation_key = keras.Input(shape = [dim_word])
entity1_id = keras.Input(shape = [dim_entity])
entity2_id = keras.Input(shape = [dim_entity])
concat_input = layers.concatenate([relation_key,entity1_id,entity2_id])
hid1 = layers.Dense(128,"tanh")(concat_input)
hid2 = layers.Dense(64,"relu")(hid1)
hid3 = layers.Dense(128,"sigmoid")(hid2)
p_correlated = layers.Dense(1,"sigmoid")(hid3)

# Construct the model of relation
Correlator = keras.Model([relation_key,entity1_id,entity2_id],p_correlated)
Correlator.summary()

key_in = keras.Input(shape = [dim_word])
entity_id = keras.Input(shape = [dim_entity])
conc_filt = layers.concatenate([key_in,entity_id])
filt1 = layers.Dense(128,"tanh")(conc_filt)
filt2 = layers.Dense(64,"relu")(filt1)
filt3 = layers.Dense(128,"sigmoid")(filt2)
prob = layers.Dense(1,"sigmoid")(filt3)

Filter = keras.Model([key_in,entity_id],prob)
Filter.summary()

