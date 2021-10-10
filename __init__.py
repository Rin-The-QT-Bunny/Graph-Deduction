import json

system_setup =  { "dim_words" : 64,
                 "dim_entity_id" : 64, 
                 'c' : 3, 'd' : 4, 'e' : 5 }

def save_setup(system_setup):
    data_setup = json.dumps(system_setup)
    f2 = open('setup.json', 'w')
    f2.write(data_setup)
    f2.close()

def load_setup():

    f = open('setup.json', 'r')
    content = f.read() # open the setup file
    settings = json.loads(content) 
    f.close() # close the setup file

    return settings

save_setup(system_setup)
