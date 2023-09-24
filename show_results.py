
from utils.lib import record_JSON
import pdb

def write_to(filename, lines):
    with open(filename, 'w') as file:
        file.write(lines)

accuracy_record = record_JSON()
accuracy_record.load()

dataset_name = 'FashionMNIST'
model_name = 'efficientnet'
combined_class=True
batch_sizes = ['single', 'batch_8', 'batch_16', 'batch', 'batch_64', 'batch_128', 'batch_256']
model_train_modes = [0,2,3,5,10]

filename = 'accuracy_record_'+dataset_name+'.csv'

accuracy_record.print_all()

lines =  'training_mode, single, batch_8, batch_16, batch_32, batch_64, batch_128, batch_256'
for model_train_mode in model_train_modes:
    lines += f'\n{model_train_mode}, '
    for batch_size in batch_sizes:    
        accuracy,idx = accuracy_record.lookup( combined_class=combined_class, 
                                model_name = model_name , 
                                model_train_mode = model_train_mode, 
                                batch_size = batch_size, 
                                dataset_name = dataset_name)
        
    #    accuracy = accuracy_record.lookup( combined_class=True, model_name = 'efficientnet', model_train_mode = 2, batch_size = 'single', dataset_name = dataset_name)
        lines += f'{accuracy}, '
    # if model_train_mode == 5:
    #     match = accuracy_record.match (model_name, combined_class, model_train_mode, 'batch', dataset_name)
    #     df = accuracy_record.get_df()
    #     accuracy = df[match]['accuracy'].values[-1]
    #     print(f'model_name = {model_name}, combined_class={combined_class}, model_train_mode= {model_train_mode}, batch_size={batch_size},dataset_name= {dataset_name},match = {match}')
    #     print(accuracy)
    #     pdb.set_trace()


print (lines)


write_to(filename, lines)