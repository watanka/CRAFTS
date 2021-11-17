import os, yaml

class Config(dict) :
    def __init__(self, config_path) :
        with open(config_path, 'r') as f :
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
            
    
    def __getattr__(self, name) :
        if self._dict.get(name) is not None :
            return self._dict[name]
        
        if DEFAULT_CONFIG.get(name) is not None :
            return DEFAULT_CONFIG[name]
        
        return None
    
    
    def print(self) :
        print('Model Configurations : ')
        print('-'*30)
        print(self._yaml)
        print('')
        print('-'*30)
        print('')
        

        
DEFAULT_CONFIG = {
    'EXP_NAME': '',                                                                     # Where to store logs and models
    'MODE' : 1,                                                                         # 1: train, 2: test
    'MODEL' : 1,                                                                        # 1: CRAFTS, 2: CRAFT(w/ orientation), 3: STR
    'SEED' : 123,                                                                       # random seed
    'GPU' : ['0'],                                                                        # list of gpu ids
    'WORKERS' : 4,                                                                      # number of data loading workers
    
    'DATA_PATH' : '/home/jovyan/nas/2_public_data/aihub_wildscene_labeled/',            # Path to data loader; should have 'image' and 'label_txt' folder
    
    'STD_CONFIG_PATH' : './config/',                                                    # Path to STD configuration file
    'STR_CONFIG_PATH' : './config/recogntion.yaml',                                     # Path to STR configuration file
    
    'LR' : 1e-5,                                                                        # initial learning rate
    'MOMENTUM' : 1e-3,                                                                  # Momentum value for optim
    'WEIGHT_DECAY' : 5e-4,                                                              # Weight decay for SGD
    'GAMMA' : 0.99,                                                                     # Gamma update for SGD
    'BATCH_SIZE' : 1,                                                                   # Batch Size
    'MAX_EPOCH' : 3000,                                                                 # Number of training teration
    
}