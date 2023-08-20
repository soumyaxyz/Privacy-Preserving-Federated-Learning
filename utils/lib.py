import sys, os, json
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

