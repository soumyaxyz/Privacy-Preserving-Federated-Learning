import matplotlib.pyplot as plt
import numpy as np
import pdb

def get_mean_std(list, j, extra_data, sublist_extra):
    data_list = [list]
    for extra in range(extra_data):
        data_list.append(sublist_extra[extra][j])
    y_mean = np.mean(data_list, axis=0)
    y_std = np.std(data_list, axis=0)

    print(f'i=? , {j=}, {list=}, {data_list=}, {y_mean=}')

    return y_mean, y_std

def plot_lists(data, datasetname, labels, placement='1x4', mean_flag=False):
    # print(labels)
    print(datasetname)

    extra_data = 0
    data_all = data
    if mean_flag:              
        extra_data = len(data)-1
        
        data = data[0]



    rows, cols = 1, len(data)  # Default is 1xN placement

    if placement == '2x2':
        rows, cols = 2, 2

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4), sharey=True)

       

       
    

    sublist_extra = []

    # pdb.set_trace()
    
    for i, sublist in enumerate(data):
        if mean_flag:
            for extra in range(extra_data):
                sublist_extra.append(data_all[extra+1][i]) 

        pdb.set_trace()
        
        

        local_labels = ["train acc","test acc", "MIA acc"]
        colours = ["tab:green", "tab:blue", "tab:red","tab:purple"]

        # mean_flag = False
        # print(len(sublist))
        # print(f'{i=} , {sublist=}') 

        if placement == '1x4':
            # pdb.set_trace()
            for j, ax in enumerate(axs):  
                             
                if j == i:
                   for k, list in enumerate(sublist):
                        if mean_flag:                            
                            try:
                                y_mean, y_std = get_mean_std(list, j, extra_data, sublist_extra)

                                
                                # print(f'{i=} , {k=}, {list=}, {data_list=}, {y_mean=}')


                                ax.plot(y_mean, label=local_labels[k], color=colours[k], marker='o', markersize=10)
                                ax.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.2, color=colours[k])
                            except Exception as e:
                                ax.plot(list, label=local_labels[k], color=colours[k])   # fallback to mean_flag = False
                        else:
                            ax.plot(list, label=local_labels[k], color=colours[k])
                        
                    
                ax.set_title(labels[j],fontsize=18)
                ax.set_yticks([60, 80, 100])               
                ax.tick_params(axis='y', labelsize=18)
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels([0, 2, 5, 10],fontsize=18)
        elif placement == '2x2':
            
                
                # for k, list in enumerate(sublist_1):
                #     try:
                #         axs[i // 2, i % 2].plot(list, label=local_labels[k], color=colours[k], marker='x', markersize=10)
                #     except:
                #         pdb.set_trace()
            

            for j, list in enumerate(sublist):
                print(f'{i=} , {j=}, {list=}')
                
                if mean_flag:
                    # print(list, sublist_1[j])
                    try:
                        y_mean, y_std = get_mean_std(list, j, extra_data, sublist_extra)

                        
                        axs[i // 2, i % 2].plot(y_mean, label=local_labels[j], color=colours[j], marker='o', markersize=10) 
                        axs[i // 2, i % 2].fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.2, color=colours[j])
                    except Exception as e:
                        print(j)
                        axs[i // 2, i % 2].plot(list, label=local_labels[j], color=colours[j], marker='o', markersize=10) # fallback to mean_flag = False
                else:
                    axs[i // 2, i % 2].plot(list, label=local_labels[j], color=colours[j], marker='o', markersize=10) 
                    # print('plotting')
                
                axs[i // 2, i % 2].set_yticks([60, 80, 100])               
                axs[i // 2, i % 2].tick_params(axis='y', labelsize=18)
                axs[i // 2, i % 2].set_title(labels[i],fontsize=18)
                axs[i // 2, i % 2].set_xticks([0, 1, 2, 3])
                axs[i // 2, i % 2].set_xticklabels([0, 2, 5, 10],fontsize=18)
    # if placement == '2x2':
    plt.legend(fontsize=15)
    
    # else:
    #     plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.show()





# Example usage

#cifar10
    # [centralized, 2, 5, 10]
fedAvg = [
    [90.57, 96.29, 95.65, 94.77], # train acc
    [83.66, 85.75, 87.04, 86.12], # test acc   
    [81.00, 82.38, 81.89, 82.47], # MIA acc

]
first = [ 
    [90.57, 90.07, 84.31, 81.31], # train acc
    [83.66, 83.2, 79.49, 77.87],    
    [81.00, 79.57, 76.83, 73.67],

]
first_1 = [ 
    [90.57, 91.13, 86.51, 84.58], # train acc
    [83.66, 82.91, 81.28, 80.24],    
    [81.00, 80.41, 77.51, 74.28],

]
confident = [
    [90.57, 84.95, 78.43, 73.92], # train acc
    [83.66, 78.9, 74.92, 71.45],    # tst acc
    [81.00, 74.99, 71.27, 66.74],   # MIA acc

]
confident_1 = [
    [90.57, 87.47, 82.15, 74.85], # train acc
    [83.66, 81.32, 78.25, 71.76],    # tst acc
    [81.00, 77.33, 74.47, 67.81],   # MIA acc

]
correct_confident = [
    [90.57, 87.07, 79.99, 78.59], # train acc
    [83.66, 81.06, 75.53, 76.15],    
    [81.00, 78.55, 72.41, 70.8],

]
correct_confident_1 = [
    [90.57, 87.72, 79.98, 75.89], # train acc
    [83.66, 80.07, 77.14, 73.33],    
    [81.00, 77.31, 72.63, 71.04],

]
round_robin = [ 
    [90.57, 89.13, 84.97, 82.79], # train acc
    [83.66, 82.01, 80.67, 78.04],    
    [81.00, 80.03, 76.53, 74.15],

]
round_robin_1 = [
    [90.57, 87.41, 84.34, 83.59], # train acc
    [83.66, 81.48, 79.16, 79.68],    
    [81.00, 78.10, 72.82, 71.99],

]

cifar10 = [ fedAvg, first,  confident, correct_confident, round_robin]

#####################################################################

#cifar10_ResNet
    # [centralized, 2, 5, 10]

fedAvg = [
    [93.9, 97.92, 94.46, 98.43], # train acc
    [80.35,80.35, 80.99, 83.07], # test acc
    [82.37, 84.68, 82.03, 84.64], # MIA acc

]
first = [
    [93.9, 96.99, 89.49 , 93.19], # train acc
    [80.35, 79.37, 76.34, 79.87],
    [82.37, 84.37, 79.65, 83.06],

]

confident = [
    [93.9, 86.59, 79.86, 68.25], # train acc
    [80.35, 74.96, 70.69, 63.95],    # tst acc
    [82.37, 77.73, 72.04, 64.39],   # MIA acc

]

correct_confident = [
    [93.9, 86.46, 74.98, 67.82], # train acc
    [80.35, 76.04, 69.86, 63.62],
    [82.37, 77.51, 68.78, 63.82],

]

round_robin = [
    [93.9, 97.14, 95.41, 89.68], # train acc
    [80.35, 79.14, 78.81, 77.41],
    [82.37, 84.46, 83.66, 80.86],

]


cifar10_ResNet = [ fedAvg, first,  confident, correct_confident, round_robin]
#####################################################################
# cifar100
# [centralized, 2, 5, 10]
fedAvg = [
    [59.45, 66.67, 83.46, 80.35], # train acc
    [50.78, 56.28, 62.54, 61.75], # test acc    
    [69.91, 72.16, 81.57, 79.34], # MIA acc

]
first = [
    [59.45, 61.01, 55.5, 39.66], # train acc
    [50.78, 50.45, 46.72, 36.31],    
    [69.91, 71.65, 66.59, 55.62],

]
first_1 = [
    [59.45, 57.91, 58.23, 47.58], # train acc
    [50.78, 49.04, 49.12, 43.03],    
    [69.91, 67.88, 71.17, 59.46],

]
confident = [
    [59.45, 57.22, 53.98, 54.8], # train acc
    [50.78, 48.2, 47.85, 48.26],    # tst acc
    [69.91, 66.93, 64.00, 66.49],   # MIA acc

]
confident_1 = [
    [59.45, 70.56, 62.62, 50.16], # train acc
    [50.78, 53.94, 51.69, 45.07],    # tst acc
    [69.91, 75.88, 71.75, 65.78],   # MIA acc

]
correct_confident = [
    [59.45, 67.35, 56.81, 49.44], # train acc
    [50.78, 54.09, 49.36, 44.78],    
    [69.91, 74.61, 66.49, 62.99],

]
correct_confident_1 = [
    [59.45, 61.61, 49.52, 52.42], # train acc
    [50.78, 52.22, 44.23, 46.94],    
    [69.91, 71.41, 63.96, 63.99],

]
round_robin = [ 
    [59.45, 54.54, 42.34, 45.36], # train acc
    [50.78, 47.1, 39.27, 40.93],    
    [69.91, 66.7, 55.76, 55.42],

]
round_robin_1 = [
    [59.45, 58.06, 53.27, 51.38], # train acc
    [50.78, 47.42, 56.57, 45.27],    
    [69.91, 68.07, 63.56, 61.28],

]

cifar100= [ fedAvg, first,  confident,  correct_confident, round_robin ]

#####################################################################


# mnist



# [centralized, 2, 5, 10]
fedAvg = [
    [99.34, 99.67, 99.75, 99.68], # train acc
    [99.08, 99.46, 99.54, 99.5],    
    [84.96, 85.12, 85.41, 85.13],
]

first = [
    [99.34, 99.51, 98.96, 98.59], # train acc
    [99.08, 99.3, 98.78, 98.36],    
    [84.96, 84.91, 84.49, 83.76],
]

confident = [
    [99.34, 99.47, 98.51, 98.42], # train acc
    [99.08, 99.31, 98.57, 98.34],    # tst acc
    [84.96, 84.86, 83.95, 84.98]  # MIA acc
]

correct_confident = [
    [99.34, 99.37, 98.83, 98.23], # train acc
    [99.08, 99.11, 98.83, 98.11],    
    [84.96, 85.07,  83.76, 81.22]
]

round_robin = [ 
    [99.34, 99.63, 99.17, 99.00], # train acc
    [99.08, 99.3, 98.75, 98.91],    # tst acc
    [84.96, 85.31, 84.46, 84.28]  # MIA acc

]


mnist = [ fedAvg, first, confident, correct_confident, round_robin]


#####################################################################
# fmnist  
# [centralized, 2, 5, 10]
fedAvg = [
    [94.93, 96.15, 96.65, 95.46], # train acc
    [92.21, 93.29, 93.48, 93.17],    
    [83.38, 84.56, 84.95, 84.61],
]

first = [
    [94.93, 96.36, 93.52, 91.84], # train acc
    [92.21, 92.79, 91.54, 90.68],    
    [83.38, 84.71, 83.63, 83.83],
]

first_1 = [
    [94.93, 95.34, 92.22, 92.00], # train acc
    [92.21, 92.59, 91.01, 90.81],    
    [83.38, 84.70, 83.73, 83.07],
]

confident = [
    [94.93, 94.17, 92.52, 92.13], # train acc
    [92.21, 91.78, 90.92, 90.22],    # tst acc
    [83.38, 83.61, 83.18, 83.74],   # MIA acc
]

confident_1 = [
    [94.93, 95.75, 93.23, 90.23], # train acc
    [92.21, 92.97, 91.43, 89.2],    # tst acc
    [83.38, 84.56, 83.45, 81.23],   # MIA acc
]

correct_confident = [
    [94.93, 95.18, 91.17, 89.9], # train acc
    [92.21, 92.86, 89.70, 89.06],    
    [83.38, 84.16, 83.06, 80.89],
]

correct_confident_1 = [
    [94.93, 95.14, 92.69, 90.12], # train acc
    [92.21, 92.63, 90.99, 89.04],    
    [83.38, 84.38, 82.22, 80.56],
]

round_robin = [ 
    [94.93, 93.44, 93.69, 91.59], # train acc
    [92.21, 91.62, 92.11, 90.27],    
    [83.38, 83.73, 83.9, 84.28],

]

round_robin_1 = [ 
    [94.93, 94.52, 92.93, 90.67], # train acc
    [92.21, 92.52, 91.3, 89.59],    
    [83.38, 83.45, 84.96, 81.12],

]

fmnist = [ fedAvg, first,  confident, correct_confident,  round_robin ]




dataset = [cifar10, cifar100, mnist, fmnist]
dataset_1 = [cifar10_ResNet, cifar100, mnist, fmnist]
datasetname = ['CIFAR 10', 'CIFAR 100', 'MNIST', 'Fashion MNIST']
labels = ['fedAvg', 'first',  'confident', 'correct_confident',  'round_robin']




# for i,data in enumerate(dataset):
#     # print(dataset[i])
#     # pdb.set_trace()
#     plot_lists([dataset[i], dataset_1[i]], datasetname[i], labels, mean_flag=True)
#     # plot_lists(dataset[i], datasetname[i], labels, mean_flag=False)
#     # break

dataset = [cifar10[0], cifar100[0] , mnist[0], fmnist[0]] 
dataset_1 = [cifar10_ResNet[0], cifar100[0] , mnist[0], fmnist[0]] 
plot_lists([dataset, dataset_1], 'Overall trends', datasetname, placement='2x2', mean_flag=True)
# plot_lists(dataset, 'Overall trends', datasetname, placement='2x2', mean_flag=False)