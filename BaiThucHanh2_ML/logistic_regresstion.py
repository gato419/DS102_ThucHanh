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
    return 1/(1+np.exp(-z))

class LogisticRegresstion:
    def  __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = np.random.random()

        for i in range(self.iterations):
            linear = X @ self.weights + self.bias
            y_predict = sigmoid(linear)
            gradient_w = 1/n_samples*X.T @ (y_predict - y)
            gradient_b = 1/n_samples * np.sum(y_predict - y)

            self.weights -= self.learning_rate*gradient_w
            self.bias -= self.learning_rate*gradient_b

            if((i+1)%100 == 0 or i==self.iterations-1):
                loss = -np.mean(y*np.log(y_predict + 1e-10) + (1 - y)*np.log(1 - y_predict + 1e-10))
                print(f"{i}    {loss}")


    def predict_prob(self, X):
        linear= X @ self.weights + self.bias
        return sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)

model = LogisticRegresstion(learning_rate=0.1, iterations=1000)
model.train(X=X_train, y=y_train)

print("Dự đoán:",model.predict(X_test[0]), "Đáp án: ",y_test[0])