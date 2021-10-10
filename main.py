"""
Main Control of C'thuen
Train the Neural Operator 
Generat the Relation Graph
"""
import setup
import utils
import word_tokenize
import graph_constructor
import neural_operators
#from tokenize import TokenMarker

# Tensorflow settings etc
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# Load the setting of the model
args = setup.load_setup()
dim_words = args["dim_words"]
max_word = 2000
max_len = 30

# Word embedder Instance
embeder_layer = layers.Embedding(max_word,dim_words)
Embeder = keras.Sequential()
Embeder.add(embeder_layer)

# Create a GRU Encoder
GRU_Single = layers.GRU(dim_words)
input_seq = keras.Input(shape = [max_len,dim_words])
g_encoder = GRU_Single(input_seq)
GEncoder = keras.Model(input_seq,g_encoder)

#Tokenizer Instance
tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')


# testing raw vocabulary material
material = ["A red ball is at the top of the building",
 "The building is next to the man",
 "The sun is burning bright",
 "The eye of Morgoth is red as the sun",
 "I saw a man shoot at a bird"]

tokenizer.fit_on_texts(material)
tokMark = word_tokenize.TokenMarker(tokenizer)
GCR = graph_constructor.GraphConstructor(Embeder,tokenizer,dim_words)



def material_entities(material):

      graph_entities = []
      #tokenizer.fit_on_texts(["A red ball is at the top of the building"])
      for k in range(len(material)):
            statement = material[k]
            #print(statement)
            Pos_Mark = tokMark.tensor_pos(statement)

                  # Create the entities shown in the statement
            Entities = GCR.construct(Pos_Mark)
            for i in range(len(Entities)):

                  graph_entities.append(Entities[i])
                  
      #print(graph_entities)
      return graph_entities

def train_property_access(pair,Mat):

      entities = material_entities(Mat)

      # pair = {entity, 'red', 1}
      # Find the embedding of the concept
      idt = tokenizer.texts_to_sequences([pair[1]])
      #print(idt)
      property_embedding = Embeder(tf.convert_to_tensor(idt))[0]
      entity_embedding = "Char"
      flag = True
      for i in range(len(entities)):
            
            # Find the embedding of the targer entity

            if pair[0] == entities[i][0]:
                  entity_embedding = entities[i][1]
                  #print(entities[i][0])
                  flag = False
      if flag:
            print("Invalid")
            print(pair[0])
            return 0
      entity_embedding = tf.reshape(entity_embedding,[1,-1])
      p = neural_operators.Filter([entity_embedding,property_embedding])[0][0]
      loss = (pair[2]-p)*(pair[2]-p)
      return loss

def Scene(material):
    
      graph_entities = []
      #tokenizer.fit_on_texts(["A red ball is at the top of the building"])
      for k in range(len(material)):
            statement = material[k]
            #print(statement)
            Pos_Mark = tokMark.tensor_pos(statement)

                  # Create the entities shown in the statement
            Entities = GCR.construct(Pos_Mark)
            for i in range(len(Entities)):

                  graph_entities.append(Entities[i])
                  
      #print(graph_entities)

      return graph_entities

def FilterProperty(Entities,Property):

      # pair = {entity, 'red', 1}
      # Find the embedding of the concept
      idt = tokenizer.texts_to_sequences([Property])
      #print(idt)
      property_embedding = Embeder(tf.convert_to_tensor(idt))[0]
      
      return_filter = []
      for i in range(len(Entities)):            
            # Find the embedding of the targer entity
            entity_embedding = Entities[i][1]
            entity_embedding = tf.reshape(entity_embedding,[1,-1])
            p = neural_operators.Filter([entity_embedding,property_embedding])[0][0]
            print(p.numpy(),Entities[i][0])
            if p>= 0.85:
                  return_filter.append(Entities[i])
      

      return return_filter

Mat = ["A red ball is at the top of the building",
 "The sun is burning bright",
 "The eye of Morgoth is red as the sun"]

def train_tasks(tasks,batch):
      """
      Format of a Task Tau：
      {
            {Materials}: Reading Material
            {Questions}:
            {Supervision}: [Entity , "Key", Bool]
      }
      {
            {Materials}: Reading Material
            {Questions}:
            {Supervision}: [Entity, "Key", Bool]
      }
      ... ... ... ... 
      """
      total_loss = 0
      for b in range(batch):
            # For Each Material
            for task_idx in range(len(tasks)):
                  # For Each question
                  for sup_idx in range(len(tasks[task_idx][2])):
                  # Cumulate the Gradient
                        supervision_pair = tasks[task_idx][2][sup_idx]
                        loss = train_property_access(supervision_pair,tasks[task_idx][0])
                        total_loss += loss
      return total_loss
            
      # Apply All Gradients to All Variables

Curriculum = [
      [
            ["A red ball is at the top of the building",
            "The sun is burning bright",
            "The eye of Morgoth is red as the sun"],

            ["Q1","Q2"],

            [["ball","red",1],["building","red",0],["sun","red",1],["morgoth","red",0]]
      ],
      [
            ["A red ball is at the top of the building",
            "The sun is burning bright",
            "The eye of Morgoth is red as the sun"],

            ["Q1","Q2"],

            [["sun","bright",1],["sun","burning",1]]
      ]
]

def save_models():
      GCR.Att.save_weights("data/Attention.h5")
      GCR.T.save_weights("data/Tranform.h5")
      Embeder.save_weights("data/Embeder.h5")
      neural_operators.Correlator.save_weights("data/Coorelator.h5")
      neural_operators.Filter.save_weights("data/Filter.h5")
def load_models():
      GCR.Att.load_weights("data/Attention.h5")
      GCR.T.load_weights("data/Tranform.h5")
      Embeder.load_weights("data/Embeder.h5")
      neural_operators.Correlator.load_weights("data/Coorelator.h5")
      neural_operators.Filter.load_weights("data/Filter.h5")



optimizer = tf.optimizers.Adam(0.02)
loss_history = []

try:
      load_models()
except:
      print("Failded to Load")


for epoch in range(1550):
      with tf.GradientTape(persistent = True) as tape:
            L = train_tasks(Curriculum,1)

      grad_a = tape.gradient(L,GCR.Att.variables)
      grad_T =  tape.gradient(L,GCR.T.variables)
      grad_e =  tape.gradient(L,Embeder.variables)

      optimizer.apply_gradients(zip(grad_a,GCR.Att.variables))
      optimizer.apply_gradients(zip(grad_T,GCR.T.variables))
      optimizer.apply_gradients(zip(grad_e,Embeder.variables))
      print("Epoch {} Loss is :".format(epoch),L.numpy())
      loss_history.append(L.numpy)
      if epoch% 10 == 0:
            save_models()





print("Saved")

all_in  = Scene(["A red ball is at the top of the building",
            "The sun is burning bright",
            "The eye of Morgoth is red as the sun"])
print("None","red")
print(FilterProperty(all_in,"red"))