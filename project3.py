import random
import math

# Sigmoid function
def sigmoid(z):
    z = max(min(z, 20), -20)  # Limit z to be within [-20, 20]
    return 1 / (1 + math.exp(-z))

# Log Loss function
def log_loss(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    return -sum(y * math.log(max(p, epsilon)) + (1 - y) * math.log(max(1 - p, epsilon)) for y, p in zip(y_true, y_pred)) / len(y_true)

# Dot product
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

# Scalar multiplication
def scalar_multiply(scalar, matrix):
    return [scalar * x for x in matrix]

# Vector subtraction
def vector_subtract(a, b):
    return [x - y for x, y in zip(a, b)]

def vector_add(a, b):
    return [x + y for x, y in zip(a, b)]


def one_hot_encode(color):
    return {'R': [1, 0, 0, 0, 0],
            'B': [0, 1, 0, 0, 0],
            'Y': [0, 0, 1, 0, 0],
            'G': [0, 0, 0, 1, 0],
            'W': [0, 0, 0, 0, 1]}[color]




def create_diagram():
    colors = ['R', 'B', 'Y', 'G']
    wire_order = []

    # Determine whether to start with a row or a column
    start_with_row = random.choice([True, False])

    diagram = [['W' for _ in range(20)] for _ in range(20)]

    for _ in range(4):
        chosen_color = random.choice(colors)
        colors.remove(chosen_color)
        wire_order.append(chosen_color)

        if start_with_row:
            row = random.randint(0, 19)
            for i in range(20):
                diagram[row][i] = chosen_color
            start_with_row = False
        else:
            col = random.randint(0, 19)
            for i in range(20):
                diagram[i][col] = chosen_color
            start_with_row = True

    arrayInput = [cell[0] for row in diagram for cell in row]  # Simplified flattening

    return diagram, wire_order, arrayInput


def get_neighborhood_feature(diagram, row, col):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= row + i < len(diagram) and 0 <= col + j < len(diagram[0]):
                neighbors.append(diagram[row + i][col + j])
    feature = {'R': 0, 'B': 0, 'Y': 0, 'G': 0, 'W': 0}
    for color in neighbors:
        feature[color[0]] += 1
    return list(feature.values())

def get_color_transition_feature(diagram, row, col):
    current_color = diagram[row][col][0]
    transitions = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= row + i < len(diagram) and 0 <= col + j < len(diagram[0]):
                neighbor_color = diagram[row + i][col + j][0]
                if current_color != neighbor_color:
                    transitions += 1
    return [transitions]


# Logistic Regression Model
class LogisticRegressionModel:
    def __init__(self, input_size, reg_lambda=0.01):
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(input_size)]
        self.bias = random.uniform(-0.1, 0.1)
        self.reg_lambda = reg_lambda  # Regularization parameter

    def predict(self, x):
        linear_output = dot_product(x, self.weights) + self.bias
        return sigmoid(linear_output)

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            total_gradient_w = [0] * len(self.weights)
            total_gradient_b = 0

            for x, y_true in zip(x_train, y_train):
                y_pred = self.predict(x)
                
                # Accumulate the log loss
                total_loss += log_loss([y_true], [y_pred])

                # Compute the gradient
                error = y_pred - y_true
                gradients_w = [x_i * error for x_i in x]
                gradient_b = error
                
                # Update gradients with L2 regularization
                gradients_w = [gw + (self.reg_lambda / len(x_train)) * w for gw, w in zip(gradients_w, self.weights)]

                # Accumulate the gradients
                total_gradient_w = vector_add(total_gradient_w, gradients_w)
                total_gradient_b += gradient_b

            # Average the gradients and the loss
            avg_gradient_w = scalar_multiply(1 / len(x_train), total_gradient_w)
            avg_gradient_b = total_gradient_b / len(x_train)
            avg_loss = total_loss / len(x_train)

            # Update weights and bias with regularization term for weights
            self.weights = vector_subtract(self.weights, scalar_multiply(learning_rate, avg_gradient_w))
            self.bias -= learning_rate * avg_gradient_b



# Generate dataset
data = []
labels = []
for _ in range(2000):  # Number of samples
    diagram, wire_order, arrayInput = create_diagram()
    feature_vector = []
    # Add one-hot encoded colors from arrayInput
    for color in arrayInput:
        feature_vector.extend(one_hot_encode(color[0]))
    for row_idx in range(len(diagram)):
        for col_idx in range(len(diagram[0])):
            feature_vector.extend(get_neighborhood_feature(diagram, row_idx, col_idx))
            feature_vector.extend(get_color_transition_feature(diagram, row_idx, col_idx))

    data.append(feature_vector)
    labels.append(1 if "R" in wire_order and "Y" in wire_order and wire_order.index("R") < wire_order.index("Y") else 0)



def split_data(data, labels, train_ratio=0.9):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    train_size = int(len(combined) * train_ratio)
    train, test = combined[:train_size], combined[train_size:]
    return train, test

train_data, test_data = split_data(data, labels)
x_train, y_train = list(zip(*train_data))  
x_test, y_test = list(zip(*test_data))    


def standardize(data):
    # Assuming data is a list of lists (each inner list is a feature vector)
    features = zip(*data)
    means = [sum(feature) / len(feature) for feature in features]
    features = zip(*data)
    stds = [math.sqrt(sum((x - mean) ** 2 for x in feature) / len(feature)) for mean, feature in zip(means, features)]
    
    standardized_data = []
    for row in data:
        standardized_row = [(x - mean) / std if std > 0 else 0 for x, mean, std in zip(row, means, stds)]
        standardized_data.append(standardized_row)
    return standardized_data

# Apply standardization to your data
x_train_scaled = standardize(x_train)
x_test_scaled = standardize(x_test)

# Train the model with scaled data
model = LogisticRegressionModel(len(x_train_scaled[0]), reg_lambda=0.01)
model.train(x_train_scaled, list(y_train), epochs=100, learning_rate=0.03)

# testing the model
correct_predictions = 0
for x, y_true in zip(x_test_scaled, y_test):
    y_pred = model.predict(x)
    correct_predictions += 1 if (y_pred > 0.5) == y_true else 0


accuracy = correct_predictions / len(x_test) * 100


# After training, calculate predictions for the test set
y_pred_test = [model.predict(x) for x in x_test]

# Calculate the average log loss on the test set
average_log_loss = log_loss(y_test, y_pred_test)

print(f"Model accuracy: {accuracy}%")

# Print the average log loss
print(f"Average Log Loss: {average_log_loss}")
