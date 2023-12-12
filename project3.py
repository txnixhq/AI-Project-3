import random
import math

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Log Loss function
def log_loss(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    return -sum(y * math.log(p + epsilon) + (1 - y) * math.log(1 - p + epsilon) for y, p in zip(y_true, y_pred)) / len(y_true)

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



def create_diagram():
    colors = ['Red', 'Blue', 'Yellow', 'Green']
    wire_order = []

    # Determine whether to start with a row or a column
    start_with_row = random.choice([True, False])

    diagram = [['White' for _ in range(20)] for _ in range(20)]

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

# Logistic Regression Model
class LogisticRegressionModel:
    def __init__(self, input_size):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)

    def predict(self, x):
        linear_output = dot_product(x, self.weights) + self.bias
        return sigmoid(linear_output)

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_gradient_w = [0] * len(self.weights)
            total_gradient_b = 0

            # Calculate the gradients for each training sample
            for x, y_true in zip(x_train, y_train):
                y_pred = self.predict(x)
                error = y_true - y_pred
                gradients_w = [x_i * error for x_i in x]
                gradient_b = error
                
                # Accumulate the gradients
                total_gradient_w = vector_add(total_gradient_w, gradients_w)
                total_gradient_b += gradient_b
            
            # Average the gradients over the dataset
            avg_gradient_w = scalar_multiply(1 / len(x_train), total_gradient_w)
            avg_gradient_b = total_gradient_b / len(x_train)

            # Update weights and bias
            self.weights = vector_add(self.weights, scalar_multiply(learning_rate, avg_gradient_w))
            self.bias += learning_rate * avg_gradient_b
def encode_color(color):
    return {'R': 1, 'B': 2, 'Y': 3, 'G': 4, 'W': 0}[color]


# Generate dataset
data = []
labels = []
for _ in range(1000):  # Number of samples
    _, wire_order, arrayInput = create_diagram()
    data.append([encode_color(color) for color in arrayInput])
    labels.append(1 if "Red" in wire_order and "Yellow" in wire_order and wire_order.index("Red") < wire_order.index("Yellow") else 0)

def split_data(data, labels, train_ratio=0.5):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    train_size = int(len(combined) * train_ratio)
    train, test = combined[:train_size], combined[train_size:]
    return train, test

train_data, test_data = split_data(data, labels)
x_train, y_train = list(zip(*train_data))  
x_test, y_test = list(zip(*test_data))    

# Train and Test the model
model = LogisticRegressionModel(400)  # 400 for a 20x20 diagram
model.train(list(x_train), list(y_train), epochs=100, learning_rate=0.01)

# Testing the model
correct_predictions = 0
for x, y_true in zip(x_test, y_test):
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
