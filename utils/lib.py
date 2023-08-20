import sys, os, json
# from IPython.utils.io import capture_output 

def blockPrinting(func):
    config_path = os.path.join('config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)  
        mode = config.get('print')
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        if mode =="all":
            pass #this decorator does nothing
        elif mode =="err":
            sys.stderr = open(os.devnull, 'w')
        elif mode =="out":
            sys.stdout = open(os.devnull, 'w')
        elif mode =="none":
            sys.stderr = open(os.devnull, 'w')
            sys.stdout = open(os.devnull, 'w')
        else:
            raise NotImplementedError
        
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value
    
    # def func_wrapper(*args, **kwargs):
    #     with capture_output():
    #         value = func(*args, **kwargs)
    #     return value

    return func_wrapper