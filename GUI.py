import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import os

def on_combobox_select(event):
    selected_value = argument_variables["Target Model Weights:"].get()
    if selected_value == "Browse for File":
        browse_for_file("saved_models", file_types=[("Model Weights", "*.pt*")])



combobox_arguments = {
    "Target Model Weights:": {
        "values": ['attack_classifier', 'another_model', 'Browse for File'],
        "on_select": on_combobox_select
    }
}


default_arguments = {
    'wandb_logging': False,
    'save_attack_model': False,
    'num_shadow_epochs': 50,
    'batchwise_loss': False,
    'combined_class': False,
    'num_attack_epochs': 50,
    'dataset_name': ['CIFAR10','CIFAR100','FashionMNIST', 'MNIST'],
    'shadow_count': 8,
    'target_model_name': ['basicCNN','efficientnet', '...'],
    'target_model_weights': ['centralizedbasicCNN', '...'],
    'shadow_model_name': ['basicCNN','efficientnet', '...'],
    'attack_model_name': ['attack_classifier', '...'],
    'distil_shadow_model': False,
    'load_attack_dataset': True,
    'save_attack_dataset': False
}

def group_variables_by_type(default_arguments):
    boolean_variables = []
    integer_variables = []
    list_variables = []
    string_variables = []

    for arg_name, default_value in default_arguments.items():
        if isinstance(default_value, bool):
            boolean_variables.append(arg_name)
        elif isinstance(default_value, int):
            integer_variables.append(arg_name)
        elif isinstance(default_value, list):
            list_variables.append(arg_name)
        else: # ASSUME isinstance(default_value, str)
            string_variables.append(arg_name)
        

    return boolean_variables, integer_variables, list_variables, string_variables

# Example usage:











# Create a function to update the combobox values based on selected values

def create_argument_widget(root, variable, widget_type, values=None, on_select=None):
    ttk.Label(root, text=variable).pack()
    if widget_type == "checkbox":
        ttk.Checkbutton(root, variable=variable).pack() #.set(initial_state)
    elif widget_type == "entry":
        ttk.Entry(root, textvariable=variable).pack()
    elif widget_type == "combobox":
        combobox = ttk.Combobox(root, textvariable=variable, values=values)
        combobox.pack()
        if on_select:
            combobox.bind("<<ComboboxSelected>>", on_select)


def run_program():
    # Implement your program logic here
    pass

def browse_for_file(initial_dir, file_types=None):
    if not file_types:
        file_types = [("All Files", "*.*")]
    else:
        file_types.append(("All Files", "*.*"))

    filename = tkinter.filedialog.askopenfilename(initialdir=initial_dir, filetypes=file_types)
    return filename

def create_gui():
    root = tk.Tk()
    root.title("Argument GUI")
    boolean_variables, integer_variables, list_variables, string_variables = group_variables_by_type(default_arguments)
    # Create Boolean arguments widgets
    for argument_name in boolean_variables:
        create_argument_widget(root, argument_name,"checkbox")

    # Create String arguments widgets
    for argument_name in integer_variables + string_variables:
        create_argument_widget(root, argument_name,  "entry")

    # Create Combobox arguments widgets
    for argument_name in string_variables:
        create_argument_widget(root, argument_name,  "combobox", combobox_config["values"], combobox_config["on_select"])

    ttk.Button(root, text="Run Program", command=run_program).pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
