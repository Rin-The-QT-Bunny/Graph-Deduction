import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


import numpy as np
import matplotlib.pyplot as plt


import random 
relation_pos = ['VERB']
entity_pos = ['NOUN','PRON','PROPN','ADJ','ADV']

# Input structure of the graph is the center ultimate prediction

class GraphConstructor:
    def __init__(self,vocab_embeder,toka,dim_word):
        self.dim_words = dim_word
        print("Graph Constructor Created")
        self.x = None
        self.word_encoder = vocab_embeder # Use the same word embeder
        self.toka = toka # Use the same tokenizer
        self.K_n = 4
        self.T,self.Att = self.create_model()
        

    def create_model(self):
        dim_words = self.dim_words
        x = keras.Input(shape = [dim_words])
        h1 = layers.Dense(100,"tanh")(x)
        h2 = layers.Dense(50,"tanh")(h1)
        h3 = layers.Dense(50,"tanh")(h2)
        y = layers.Dense(dim_words,"tanh")(h3)
        T =keras.Model(x,y)

        node0 = keras.Input(shape = [dim_words])
        node1 = keras.Input(shape = [dim_words])
        nodes_concat = layers.concatenate([node0,node1])
        hidden = layers.Dense(16,"tanh")(nodes_concat)
        act = layers.Dense(1,"sigmoid")(hidden)
        att = keras.Model([node0,node1],act)
        #print("GET is initiated!")
        return T, att


    def construct(self,x):
        # Input x of the constructor is in the form of
        # [
        # 
        # [19, 'PRON'], [20, 'VERB'], [3, 'DET'], [8, 'NOUN'], [21, 'VERB'], [5, 'ADP'], [3, 'DET'], [22, 'NOUN']
        # 
        # ]
        entity_representations = list()
        #actual text is also calculated for convenience#

        # transfer to word into sequence form : [ 1, 6, 7, 15, 17, 4, 9]
        word_sequence = []
        for i in range(len(x)):
            word_sequence.append(x[i][0])
        word_sequence = tf.convert_to_tensor(word_sequence)

        # Every word atom is assigned an vector representation
        word_vectors = self.word_encoder(word_sequence)

        # actual words these representations correspond to 
        word_text = list()
        for i in range(len(x)):
          word_atom = self.toka.sequences_to_texts([x[i:i+1][0]]) 
          word_text.append(word_atom)    

        # Generate entity representation for all entites using a GAT
        for i in range(len(x)):

            # Check the pos of the word under the context 
            is_entity = x[i][1] in entity_pos

            # if it is an entity, then create a embedding for that
            entity_raw_vec = 0
            if is_entity:
                #print("Found Entity",x[i][0],word_text[i])
                for j in range(len(x)):
                    
                    word_i = tf.convert_to_tensor(word_vectors[i])
                    word_j = tf.convert_to_tensor(word_vectors[j])

                    word_i = tf.reshape(word_i,[1,self.dim_words])
                    word_j = tf.reshape(word_j,[1,self.dim_words])

                    att = self.Att([word_i,word_j])[0]
                    entity_raw_vec += att * self.T(word_vectors[j:j+1])[0]
                entity_representations.append([word_text[i][0], entity_raw_vec])
        # Going to create a list of {Word_text,Vector}
        return entity_representations
    


