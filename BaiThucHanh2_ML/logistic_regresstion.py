import numpy as np

TRAIN_IMAGE = 'train-images.idx3-ubyte'
TRAIN_LABEL = 'train-labels.idx1-ubyte'
TEST_IMAGE = 't10k-images.idx3-ubyte'
TEST_LABEL = 't10k-labels.idx1-ubyte'

def load_images(filepath):
    with open(filepath, 'rb') as f:
        # Bỏ qua 16 bytes header (Magic number, number of images, rows, cols)
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    # Reshape về dạng (Số ảnh, 784 điểm ảnh)
    return data.reshape(-1, 28 * 28)

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        # Bỏ qua 8 bytes header (Magic number, number of items)
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

X_train_raw = load_images(TRAIN_IMAGE)
y_train_raw = load_labels(TRAIN_LABEL)
X_test_raw = load_images(TEST_IMAGE)
y_test_raw = load_labels(TEST_LABEL)

X_train = X_train_raw[(y_train_raw == 0) | (y_train_raw == 1)] / 255.0 
y_train = y_train_raw[(y_train_raw == 0) | (y_train_raw == 1)]

X_test = X_test_raw[(y_test_raw == 0) | (y_test_raw == 1)] / 255.0 
y_test = y_test_raw[(y_test_raw == 0) | (y_test_raw == 1)]

print(f"Kích thước tập Train (chỉ gồm 0 & 1): {X_train.shape}")
print(f"Kích thước tập Test  (chỉ gồm 0 & 1): {X_test.shape}\n")

def sigmoid(z):
    return 1/np.exp(-z)