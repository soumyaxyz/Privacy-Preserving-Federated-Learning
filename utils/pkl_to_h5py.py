import h5py
import torch

def save_dataset_hdf5(train_dataset, test_dataset, num_channels, num_classes, filename):
    # Open an HDF5 file
    with h5py.File(filename, 'w') as f:
        # Saving datasets
        for name, dataset in [("train", train_dataset), ("test", test_dataset)]:
            images = []
            labels = []
            # Assuming dataset is a PyTorch Dataset
            for img, label in dataset:
                img = img.numpy()  # Convert PyTorch tensor to numpy array
                images.append(img)
                labels.append(label)
            images = np.stack(images)
            labels = np.array(labels)
            # Create datasets for images and labels within the HDF5 file
            f.create_dataset(f'{name}/images', data=images)
            f.create_dataset(f'{name}/labels', data=labels)
        # Saving additional info
        f.attrs['num_channels'] = num_channels
        f.attrs['num_classes'] = num_classes

def load_dataset_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        # Loading datasets
        train_images = torch.tensor(f['train/images'][:])
        train_labels = torch.tensor(f['train/labels'][:], dtype=torch.long)
        test_images = torch.tensor(f['test/images'][:])
        test_labels = torch.tensor(f['test/labels'][:], dtype=torch.long)
        # Creating PyTorch datasets from tensors
        train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
        # Loading additional info
        num_channels = f.attrs['num_channels']
        num_classes = f.attrs['num_classes']
    return train_dataset, test_dataset, num_channels, num_classes
