import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns
import pdb, traceback

from utils.lib import blockPrintingIfServer



def get_confidence(prediction, filtered = False, value=1):
    (confidence, eval_results) = prediction # type: ignore   
    if filtered:
        filtered_confidence = confidence[eval_results == value]
        confidence = filtered_confidence 

    # confidence = torch.nn.functional.softmax(confidence, dim=1)
    # pdb.set_trace()

    confidence = sorted(confidence)
    return confidence

def plot_histogram(predTrain, predTest):
    dpi_value = 100
    fig, axs = plt.subplots(1, 3, figsize=(15, 3), dpi=dpi_value)  # 1 row, 3 columns
    title = ['All', 'Correctly Classified', 'Incorrectly Classified']

    # Set a common range for the x-axis
    x_axis_range = (0, 1)

    # Initialize variables to track the maximum y-axis limit
    max_y_limit = 0

    for i, ax in enumerate(axs):
        if i == 0:
            trn_conf = get_confidence(predTrain, filtered=False, value=1)
            tst_conf = get_confidence(predTest, filtered=False, value=1)
        elif i == 1:
            trn_conf = get_confidence(predTrain, filtered=True, value=1)
            tst_conf = get_confidence(predTest, filtered=True, value=1)
        else:
            trn_conf = get_confidence(predTrain, filtered=True, value=0)
            tst_conf = get_confidence(predTest, filtered=True, value=0)

        # ax.title.set_text(title[i])

        # Update the maximum y-axis limit
        max_y_limit = max(max_y_limit, max(ax.hist(trn_conf, bins=20, range=x_axis_range, alpha=0.5, label='train', edgecolor='black')[0]))

        max_y_limit = max(max_y_limit, max(ax.hist(tst_conf, bins=20, range=x_axis_range, alpha=0.5, label='test', edgecolor='black')[0]))
        
        ax.set_xlabel(title[i], fontsize=18)
        ax.set_ylabel('Sample count', fontsize=18)
        ax.legend(fontsize=18)

    # Set the same y-axis limits for all subplots
    for ax in axs:
        ax.set_ylim(0, max_y_limit*1.05)

    plt.tight_layout()
    plt.show()



def unnormalize(image, transform):
    # Assuming the last step in the transform is Normalize
    # Extract the mean and std directly from the transform
    for t in transform.transforms:
        if isinstance(t, transforms.Normalize):
            mean = torch.tensor(t.mean).view(3, 1, 1)
            std = torch.tensor(t.std).view(3, 1, 1)
            break
    
    image = image * std + mean  # Unnormalize
    image = image.clamp(0, 1)  # Ensure values are within [0, 1]
    return image

def show_img(data_point, transform):

    image, label = data_point

    image = unnormalize(image, transform)
    
    image = image.numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))

    # Display the plot
    plt.title(f'Label: {label}')
    plt.show()

def build_ROC(plt, gold, pred, log_log = False):
    try:
        fpr, tpr, thresholds = roc_curve(gold, pred)
        roc_auc = roc_auc_score(gold, pred)
        if log_log:
            # Plot the ROC curve in log-log scale    
            plt.loglog(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.loglog([1e-6, 1], [1e-6, 1], color='gray', linestyle='--')
            plt.xlim([1e-6, 1])
            plt.ylim([1e-6, 1])
            plt.xlabel('False Positive Rate (log)')
            plt.ylabel('True Positive Rate (log)')
            plt.title('ROC Curve (log-log scale)')
        else:
            # Plot the ROC curve            
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()

@blockPrintingIfServer
def plot_ROC_curve(gold, pred, log_log = False):
    plt.figure(figsize=(8, 6))
    build_ROC(plt, gold, pred, log_log)   

    plt.legend(loc='lower right')
    plt.show()

    plot_CM(gold, pred)

@blockPrintingIfServer
def plot_CM(gold, pred):
    # Compute confusion matrix
    cm = confusion_matrix(gold, pred)

    # Create a heatmap of the confusion matrix
    class_names = ['Negative', 'Positive']  # Replace with your class names
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names) # type: ignore
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()