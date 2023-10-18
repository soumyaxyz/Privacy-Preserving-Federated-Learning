import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import os

# Define a list of argument names
argument_names = [
    "Enable Wandb Logging:",
    "Save Attack Model:",
    "Number of Shadow Epochs:",
    "Batchwise Loss:",
    "Combined Class:",
    "Number of Attack Epochs:",
    "Dataset Name:",
    "Shadow Count:",
    "Target Model Name:",
    "Target Model Weights:",
    "Shadow Model Name:",
    "Attack Model Name:",
    "Distil Shadow Model:",
    "Load Attack Dataset:",
    "Save Attack Dataset:"
]

# Create a dictionary to hold the variable types for each argument
argument_variable_types = {
    "Enable Wandb Logging:": tk.BooleanVar,
    "Save Attack Model:": tk.BooleanVar,
    "Number of Shadow Epochs:": tk.StringVar,
    "Batchwise Loss:": tk.BooleanVar,
    "Combined Class:": tk.BooleanVar,
    "Number of Attack Epochs:": tk.StringVar,
    "Dataset Name:": tk.StringVar,
    "Shadow Count:": tk.StringVar,
    "Target Model Name:": tk.StringVar,
    "Target Model Weights:": tk.StringVar,
    "Shadow Model Name:": tk.StringVar,
    "Attack Model Name:": tk.StringVar,
    "Distil Shadow Model:": tk.BooleanVar,
    "Load Attack Dataset:": tk.BooleanVar,
    "Save Attack Dataset:": tk.BooleanVar
}

# Create a dictionary to hold the default values for certain arguments
default_argument_values = {
    "Number of Shadow Epochs:": "50",
    "Number of Attack Epochs:": "50",
    "Dataset Name:": "CIFAR10",
    "Shadow Count:": "8",
    "Target Model Name:": "basicCNN",
    "Target Model Weights:": "centralizedbasicCNN",
    "Shadow Model Name:": "basicCNN",
    "Attack Model Name:": "attack_classifier",
    "Distil Shadow Model:": False,
    "Load Attack Dataset:": True
}

# Create a dictionary to hold the combobox values and associated function for specific arguments
combobox_arguments = {
    "Target Model Weights:": {
        "values": ['attack_classifier', 'another_model', 'Browse for File'],
        "on_select": on_combobox_select
    }
}

# Create a dictionary to hold the variables
argument_variables = {}

# Create a function to update the combobox values based on selected values
def on_combobox_select(event):
    selected_value = argument_variables["Target Model Weights:"].get()
    if selected_value == "Browse for File":
        browse_for_file("saved_models", file_types=[("Model Weights", "*.pt*")])

def create_gui():
    root = tk.Tk()
    root.title("Argument GUI")

    for argument_name in argument_names:
        variable_type = argument_variable_types[argument_name]
        default_value = default_argument_values.get(argument_name, "")
        variable = variable_type(value=default_value)
        argument_variables[argument_name] = variable

        if argument_name in combobox_arguments:
            values = combobox_arguments[argument_name]["values"]
            on_select = combobox_arguments[argument_name]["on_select"]
            create_argument_widget(root, argument_name, variable, "combobox", values, on_select)
        else:
            create_argument_widget(root, argument_name, variable, "entry")

    ttk.Button(root, text="Run Program", command=run_program).pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()









# import argparse
# import tkinter as tk
# from tkinter import ttk
# import tkinter.filedialog
# import os

# def parse_args(args_dict):
#     parser = argparse.ArgumentParser(description='A description of your program')
#     parser.add_argument('-w', '--wandb_logging', action='store_true', help='Enable wandb logging')
#     parser.add_argument('-sm', '--save_attack_model', action='store_true', help='save the developed attack model')
#     parser.add_argument('-e', '--num_shadow_epochs', type=int, default=50, help='Number of rounds of shadow training')
#     parser.add_argument('-b', '--batchwise_loss', action='store_true', help='For attack model training, loss is calculated batchwise, otherwise loss is calculated samplewise')
#     parser.add_argument('-c', '--combined_class', action='store_true', help='if this flag is present, combined class attack, otherwise classwise separate attack')
#     parser.add_argument('-e1', '--num_attack_epochs', type=int, default=50, help='Number of rounds of attack training')
#     parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name, target dataset')
#     parser.add_argument('-n', '--shadow_count', type=int, default=8, help='Number of shadow models')
#     parser.add_argument('-m', '--target_model_name', type=str, default='basicCNN', help='Model name for the model to be attacked')
#     parser.add_argument('-mw', '--target_model_weights', type=str, default='centralizedbasicCNN', help='Weights for the model to be attacked')
#     parser.add_argument('-s', '--shadow_model_name', type=str, default='basicCNN', help='Model name for the shadow model')
#     parser.add_argument('-a', '--attack_model_name', type=str, default='attack_classifier', help='Classifier for the attack model')
#     parser.add_argument('-ds', '--distil_shadow_model', action='store_true', help='For shadow model training, the shadow models are distilled from the target model, otherwise the shadow models are trained from scratch')
    
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument('-ld', '--load_attack_dataset', action='store_true', help='Instead of building the attack dataset, load a pre-existing attack dataset from disc')
#     group.add_argument('-sv', '--save_attack_dataset', action='store_true', help='Save the computed attack dataset to disc')

#     args = parser.parse_args(args=args_dict)
#     return args

# def run_program():
#     # Create a dictionary to hold the values from the GUI widgets
#     args_dict = {
#         'wandb_logging': wandb_logging_var.get(),
#         'save_attack_model': save_attack_model_var.get(),
#         'num_shadow_epochs': num_shadow_epochs_var.get(),
#         'batchwise_loss': batchwise_loss_var.get(),
#         'combined_class': combined_class_var.get(),
#         'num_attack_epochs': num_attack_epochs_var.get(),
#         'dataset_name': dataset_name_var.get(),
#         'shadow_count': shadow_count_var.get(),
#         'target_model_name': target_model_name_var.get(),
#         'target_model_weights': target_model_weights_var.get(),
#         'shadow_model_name': shadow_model_name_var.get(),
#         'attack_model_name': attack_model_name_var.get(),
#         'distil_shadow_model': distil_shadow_model_var.get(),
#         'load_attack_dataset': load_attack_dataset_var.get(),
#         'save_attack_dataset': save_attack_dataset_var.get()
#     }

#     # Call your script's main function with the parsed arguments
#     args = parse_args(args_dict)
#     main(args)

def relative_path_to_folder(folder_name):
    current_dir = os.getcwd()  # Get the current working directory
    return os.path.join(current_dir, folder_name)  # Replace "relative_path_to_folder" with your desired relative folder path

def browse_for_file(initial_dir ='', file_types = None): 
    if  file_types is None:
        file_types = [("Text Files", "*.txt")]
    
    all_files_option = ("All Files", "*.*")
    file_types.append(all_files_option) # type: ignore

    return tkinter.filedialog.askopenfilename(initialdir=initial_dir, filetypes=file_types)


# # def create_gui():
#     root = tk.Tk()
#     root.title("Argument GUI")

#     # Create labels and widgets for each argument
#     ttk.Label(root, text="Enable Wandb Logging:").pack()
#     wandb_logging_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=wandb_logging_var).pack()

#     ttk.Label(root, text="Save Attack Model:").pack()
#     save_attack_model_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=save_attack_model_var).pack()

#     ttk.Label(root, text="Number of Shadow Epochs:").pack()
#     num_shadow_epochs_var = tk.StringVar()
#     num_shadow_epochs_entry = ttk.Entry(root, textvariable=num_shadow_epochs_var)
#     num_shadow_epochs_entry.pack()

#     ttk.Label(root, text="Batchwise Loss:").pack()
#     batchwise_loss_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=batchwise_loss_var).pack()

#     ttk.Label(root, text="Combined Class:").pack()
#     combined_class_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=combined_class_var).pack()

#     ttk.Label(root, text="Number of Attack Epochs:").pack()
#     num_attack_epochs_var = tk.StringVar()
#     num_attack_epochs_entry = ttk.Entry(root, textvariable=num_attack_epochs_var)
#     num_attack_epochs_entry.pack()

#     ttk.Label(root, text="Dataset Name:").pack()
#     dataset_name_var = tk.StringVar()
#     dataset_name_entry = ttk.Entry(root, textvariable=dataset_name_var)
#     dataset_name_entry.pack()

#     ttk.Label(root, text="Shadow Count:").pack()
#     shadow_count_var = tk.StringVar()
#     shadow_count_entry = ttk.Entry(root, textvariable=shadow_count_var)
#     shadow_count_entry.pack()

#     ttk.Label(root, text="Target Model Name:").pack()
#     target_model_name_var = tk.StringVar()
#     target_model_name_entry = ttk.Entry(root, textvariable=target_model_name_var)
#     target_model_name_entry.pack()

#     ttk.Label(root, text="Target Model Weights:").pack()
#     target_model_weight_var = tk.StringVar()
#     Target_Model_Weights = ttk.Combobox(root, textvariable=target_model_weight_var, values=['attack_classifier', 'another_model', 'Browse for File'])
#     Target_Model_Weights.pack()
#     def on_combobox_select(event):
#         if target_model_weight_var.get() == "Browse for File":                       
#             target_model_weight_var.set(browse_for_file(relative_path_to_folder("saved_models"), file_types=[("Model Weights", "*.pt*")]) )
            
#     Target_Model_Weights.bind("<<ComboboxSelected>>", on_combobox_select)

#     ttk.Label(root, text="Shadow Model Name:").pack()
#     shadow_model_name_var = tk.StringVar()
#     shadow_model_name_entry = ttk.Entry(root, textvariable=shadow_model_name_var)
#     shadow_model_name_entry.pack()

    
#     ttk.Label(root, text="Attack Model Name:").pack()
#     attack_model_name_var = tk.StringVar()
#     target_model_weights_entry = ttk.Entry(root, textvariable=attack_model_name_var)
#     target_model_weights_entry.pack()
    

#     ttk.Label(root, text="Distil Shadow Model:").pack()
#     distil_shadow_model_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=distil_shadow_model_var).pack()

#     ttk.Label(root, text="Load Attack Dataset:").pack()
#     load_attack_dataset_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=load_attack_dataset_var).pack()

#     ttk.Label(root, text="Save Attack Dataset:").pack()
#     save_attack_dataset_var = tk.BooleanVar()
#     ttk.Checkbutton(root, variable=save_attack_dataset_var).pack()

#     ttk.Button(root, text="Run Program", command=run_program).pack()

#     root.mainloop()

def create_argument_widget(root, label_text, variable, widget_type, values=None, on_select=None):
    ttk.Label(root, text=label_text).pack()
    if widget_type == "checkbutton":
        widget = ttk.Checkbutton(root, variable=variable)
    elif widget_type == "entry":
        widget = ttk.Entry(root, textvariable=variable)
    elif widget_type == "combobox":
        widget = ttk.Combobox(root, textvariable=variable, values=values) # type: ignore
        if on_select:
            widget.bind("<<ComboboxSelected>>", on_select)
    widget.pack()
    
# # def on_combobox_select(event, variable):
# #     if variable.get() == "Browse for File":
# #         browse_for_file("saved_models", file_types=[("Model Weights", "*.pt*")])

# def on_combobox_select(selected_value):
#     if selected_value == "Browse for File":
#         browse_for_file("saved_models", file_types=[("Model Weights", "*.pt*")])
#     return selected_value 

# # def on_combobox_select(event):
# #     selected_value = target_model_weight_var.get()
# #     if selected_value == "Browse for File":
# #         browse_for_file("saved_models", file_types=[("Model Weights", "*.pt*")])




# def create_gui():
#     root = tk.Tk()
#     root.title("Argument GUI")

#     wandb_logging_var = tk.BooleanVar()
#     save_attack_model_var = tk.BooleanVar()
#     num_shadow_epochs_var = tk.StringVar()
#     batchwise_loss_var = tk.BooleanVar()
#     combined_class_var = tk.BooleanVar()
#     num_attack_epochs_var = tk.StringVar()
#     dataset_name_var = tk.StringVar()
#     shadow_count_var = tk.StringVar()
#     target_model_name_var = tk.StringVar()
#     target_model_weight_var = tk.StringVar()
#     shadow_model_name_var = tk.StringVar()
#     attack_model_name_var = tk.StringVar()
#     distil_shadow_model_var = tk.BooleanVar()
#     load_attack_dataset_var = tk.BooleanVar()
#     save_attack_dataset_var = tk.BooleanVar()

#     create_argument_widget(root, "Enable Wandb Logging:", wandb_logging_var, "checkbutton")
#     create_argument_widget(root, "Save Attack Model:", save_attack_model_var, "checkbutton")
#     create_argument_widget(root, "Number of Shadow Epochs:", num_shadow_epochs_var, "entry")
#     create_argument_widget(root, "Batchwise Loss:", batchwise_loss_var, "checkbutton")
#     create_argument_widget(root, "Combined Class:", combined_class_var, "checkbutton")
#     create_argument_widget(root, "Number of Attack Epochs:", num_attack_epochs_var, "entry")
#     create_argument_widget(root, "Dataset Name:", dataset_name_var, "entry")
#     create_argument_widget(root, "Shadow Count:", shadow_count_var, "entry")
#     create_argument_widget(root, "Target Model Name:", target_model_name_var, "entry")
#     # create_argument_widget(root, "Target Model Weights:", target_model_weight_var, "combobox", values=['attack_classifier', 'another_model', 'Browse for File'], on_select=lambda event: on_combobox_select(event, target_model_weight_var))
#     create_argument_widget(root, "Target Model Weights:", target_model_weight_var, "combobox", values=['attack_classifier', 'another_model', 'Browse for File'],  
#         # on_select=lambda event, var=target_model_weight_var: on_combobox_select(var.get())
#         on_select=lambda event, var=target_model_weight_var: target_model_weight_var.set(on_combobox_select(var.get())) )
    
#     Target_Model_Weights = ttk.Combobox(root, textvariable=target_model_weight_var, values=['attack_classifier', 'another_model', 'Browse for File'])
#     Target_Model_Weights.pack()
#     def on_combobox_select(event):
#         if target_model_weight_var.get() == "Browse for File":                       
#             target_model_weight_var.set(browse_for_file(relative_path_to_folder("saved_models"), file_types=[("Model Weights", "*.pt*")]) )
            
#     Target_Model_Weights.bind("<<ComboboxSelected>>", on_combobox_select)



#     create_argument_widget(root, "Shadow Model Name:", shadow_model_name_var, "entry")
#     create_argument_widget(root, "Attack Model Name:", attack_model_name_var, "entry")
#     create_argument_widget(root, "Distil Shadow Model:", distil_shadow_model_var, "checkbutton")
#     create_argument_widget(root, "Load Attack Dataset:", load_attack_dataset_var, "checkbutton")
#     create_argument_widget(root, "Save Attack Dataset:", save_attack_dataset_var, "checkbutton")

#     ttk.Button(root, text="Run Program", command=run_program).pack()

#     root.mainloop()

# if __name__ == "__main__":
#     create_gui()
