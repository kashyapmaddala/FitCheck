import tensorflow as tf
import tensorflow_datasets as tfds

import pickle as pkl

# # Extracting the Fashion MNIST dataset using TensorFlow Datasets
# data, metadata = tfds.load('fashion_mnist', with_info=True, as_supervised=True)

# # Uploading the dataset to a pickle file
# with open('fashion_mnist.pkl', 'wb') as f:
#     pkl.dump(data, f)

with open('fashion_mnist.pkl', 'rb') as f:
    data = pkl.load(f)

print(data)