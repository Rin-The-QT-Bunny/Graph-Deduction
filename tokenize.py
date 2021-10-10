import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import sklearn as sk
import numpy as np
import spacy

import tensorflow as tf

import setup # setting of the model is loaded again

# set up the kind of pos that represents entities and relationss
relation_pos = ['VERB'] # considered as relations words
entity_pos = ['NOUN','PRON','PROPN','ADJ','ADV'] # these will be selected


# Create a pos marking class
class TokenMarker:
    
    def __init__(self,tok):
        print("TokenMarker is Created")
        self.nlp = spacy.load('en_core_web_sm') 
        self.K = 5
        self.tokenizer = tok

    def word_pos(self,word):
        prep = self.nlp(word)
        for token in prep:
            return token.pos_
    def last_pos(self,word):
        prep = self.nlp(word)
        tok_list = []
        for tok in prep:
            tok_list.append(tok.pos_)
        return tok_list[-1]

    def mark_pos(self,x):
        text = [x]
        sequence = self.tokenizer.texts_to_sequences(text)
        pos_list = []
        for i in range(len(sequence[0])):
            # First, locate the single token
            word_atom = self.tokenizer.sequences_to_texts([sequence[0][i:i+1]])[0]
            partion = sequence[0][i:]
            if len(partion) >self.K:
                context = self.tokenizer.sequences_to_texts([partion])
                pos_list.append(self.word_pos(context[0]))
            else:
                partion = sequence[0][:i]
                context = self.tokenizer.sequences_to_texts([sequence[0][0:i+1]])
                pos_list.append(self.last_pos(context[0]))
            
        return_seq = list()
        for i in range(len(pos_list)):
            return_seq.append(pos_list[i])
        return return_seq

    def tensor_pos(self,x):
        """
        Input Format:
        [
            [4], [5], [6], [2], [7], [1], [8], [9], [1], [3]
        ]

        Hidden Format:
        [
            [A], [red], [ball], [is], [at], [the]
        ]

        Output Format:
        [
            [4, NOUN], [5, ADP], [6, DET], [2, NOUN], [7, NOUN]
        ]
        """
        text = [x]
        sequence = self.tokenizer.texts_to_sequences(text)
        pos_list = []
        for i in range(len(sequence[0])):
            # First, locate the single token
            word_atom = self.tokenizer.sequences_to_texts([sequence[0][i:i+1]])[0]
            partion = sequence[0][i:]
            if len(partion) >self.K:
                context = self.tokenizer.sequences_to_texts([partion])
                pos_list.append(self.word_pos(context[0]))
            else:
                partion = sequence[0][:i]
                context = self.tokenizer.sequences_to_texts([sequence[0][0:i+1]])
                pos_list.append(self.last_pos(context[0]))
            
        return_seq = list()
        for i in range(len(pos_list)):
            return_seq.append([sequence[0][i],pos_list[i]])
        return return_seq
            # Second. tra nsfer the K context 

            # Mark the token with pos_ tag_

        # Return the series
