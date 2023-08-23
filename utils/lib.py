import sys, os, json
import pandas as pd
import pdb, traceback
import pprint
# from IPython.utils.io import capture_output # commented version for jupyter notebook

# def modify_output(function, *args, **kwargs):
#     with capture_output():
#         value = function(*args, **kwargs)
#     return value

def modify_output(mode, target, function, *args, **kwargs):
    if mode =="all":
        pass #this decorator does nothing
    elif mode =="err":
        sys.stdout = open(target, 'w')
    elif mode =="out":
        sys.stderr = open(target, 'w')
    elif mode =="none":
        sys.stderr = open(target, 'w')
        sys.stdout = open(target, 'w')
    else:
        raise NotImplementedError
    
    # call the method in question
    value = function(*args, **kwargs)
    # enable all printing to the console
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    # pass the return value of the method back
    return value

def load_from_config(serveMode=False):
    config_path = os.path.join('config.json')
    with open(config_path) as config_file:
        config  = json.load(config_file)
        mode    = config.get('print')
        target  = config.get('redirect_to')         
        if not target:
            target = os.devnull 
        if serveMode:      
            server = config.get('server')
            if server=="true":
                mode = "none"
            else:
                mode = "all"

    return mode, target

def blockPrinting(function):
    mode, target = load_from_config()
    def function_wrapper(*args, **kwargs):
        return modify_output(mode, target, function, *args, **kwargs)
    return function_wrapper

def blockPrintingIfServer(function):
    mode, target = load_from_config(serveMode=True)
    def function_wrapper(*args, **kwargs):
        return modify_output(mode, target, function, *args, **kwargs)
    return function_wrapper

def create_directories_if_not_exist(path):
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    except:
        traceback.print_exc()
        pdb.set_trace()


class record_JSON():
    def __init__(self, filename='accuracy_record.json') -> None:
        create_directories_if_not_exist(filename)
        self.filename = filename    

        self.record_pd = pd.DataFrame(columns=[ 'combined_class', 'model_name', 'model_train_mode', 'batch_size', 'dataset_name', 'accuracy']) 
        self.loaded = False

    def save(self):
        record_json = self.record_pd.to_json(orient='records') 
        file_path = self.filename
        with open(file_path, 'w') as json_file:
            json_file.write(record_json)

    def load(self):
        with open(self.filename, 'r') as json_file:
            self.record_pd = pd.read_json(json_file)

    def print_all(self):
        if not self.loaded:
            self.load()
        pprint.pprint(self.record_pd)

    def record(self, accuracy, combined_class, model_name, model_train_mode, batch_size, dataset_name):
        self.record_pd.loc[len(self.record_pd.index)] = [combined_class, model_name, model_train_mode, batch_size, dataset_name, accuracy]  # type: ignore

    def lookup(self, combined_class=True, model_name='efficientnet', model_train_mode = 0, batch_size = '32', dataset_name = 'CIFAR10') -> float:
        if not self.loaded:
            self.load()
        match = (
            (self.record_pd['model_name'] == model_name) &
            (self.record_pd['combined_class'] == combined_class) &
            (self.record_pd['model_train_mode'] == model_train_mode) &
            (self.record_pd['batch_size'] == batch_size) &
            (self.record_pd['dataset_name'] == dataset_name)
            )
        try:
            accuracy = self.record_pd[match]['accuracy'].values[-1]
        except:
            # pdb.set_trace()
            accuracy = 0
        
        return accuracy





 # will have to be a global variable since its value will continuously change and will need to be stored

