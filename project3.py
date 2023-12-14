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

#log loss for task2
def multi_class_log_loss(y_true, y_pred):
    epsilon = 1e-15
    loss = 0
    for y_t, y_p in zip(y_true, y_pred):
        loss += -sum(y * math.log(max(p, epsilon)) for y, p in zip(y_t, y_p))
    return loss / len(y_true)


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

#for the softmax model
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(i - max_logit) for i in logits]
    sum_of_exps = sum(exps)
    return [j / sum_of_exps for j in exps]



def one_hot_encode(color):
    return {'R': [1, 0, 0, 0, 0],
            'B': [0, 1, 0, 0, 0],
            'Y': [0, 0, 1, 0, 0],
            'G': [0, 0, 0, 1, 0],
            'W': [0, 0, 0, 0, 1]}[color]



# THIS IS FOR TASK1
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


#THIS IS FOR TASK 2
def create_dangerous_diagram():
    colors = ['R', 'B', 'Y', 'G']
    random.shuffle(colors)

    # Ensure the diagram is dangerous ('R' comes before 'Y')
    while colors.index('R') > colors.index('Y'):
        random.shuffle(colors)

    start_with_row = random.choice([True, False])
    diagram = [['W' for _ in range(20)] for _ in range(20)]

    for color in colors:
        if start_with_row:
            row = random.randint(0, 19)
            for i in range(20):
                diagram[row][i] = color
            start_with_row = False
        else:
            col = random.randint(0, 19)
            for i in range(20):
                diagram[i][col] = color
            start_with_row = True

    arrayInput = [cell for row in diagram for cell in row]

    # Determine the wire to cut based on your project's specific rule
    # For example, cutting the third wire laid down
    wire_to_cut = colors[2]

    return diagram, colors, arrayInput, wire_to_cut



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

        # Print average loss every epoch
            avg_loss = total_loss / len(x_train)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Loss = {avg_loss}")

#Softmax for task 2
class SoftmaxRegressionModel:
    def __init__(self, input_size, num_classes, reg_lambda=0.01):
        # Initialize weights and bias
        self.weights = [[random.uniform(-0.01, 0.01) for _ in range(num_classes)] for _ in range(input_size)]
        self.bias = [0.0 for _ in range(num_classes)]
        self.reg_lambda = reg_lambda

    def predict(self, x):
        # Compute logits and apply softmax
        logits = [sum(x[i] * self.weights[i][j] for i in range(len(x))) + self.bias[j] for j in range(len(self.bias))]
        return softmax(logits)

    def cross_entropy_loss(self, y_true, y_pred):
        return -sum(y_true[i] * math.log(y_pred[i] + 1e-15) for i in range(len(y_true)))


    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0

            for x, y_true in zip(x_train, y_train):
                y_pred = self.predict(x)

                # Compute error and loss
                error = [y_pred[i] - y_true[i] for i in range(len(y_true))]
                total_loss += self.cross_entropy_loss(y_true, y_pred)

                # Compute gradients
                gradients_w = [[x[i] * error[j] for j in range(len(error))] for i in range(len(x))]
                gradient_b = error

                # Update weights and bias with regularization
                self.weights = [[self.weights[i][j] - learning_rate * (gradients_w[i][j] + self.reg_lambda * self.weights[i][j]) for j in range(len(self.weights[0]))] for i in range(len(self.weights))]
                self.bias = [self.bias[j] - learning_rate * gradient_b[j] for j in range(len(self.bias))]

            # Print average loss every epoch
            avg_loss = total_loss / len(x_train)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Loss = {avg_loss}")

    def evaluate(self, x_test, y_test):
        correct_predictions = 0
        for x, y_true in zip(x_test, y_test):
            y_pred = self.predict(x)
            predicted_class = y_pred.index(max(y_pred))
            true_class = y_true.index(max(y_true))
            if predicted_class == true_class:
                correct_predictions += 1

        return correct_predictions / len(x_test) * 100


def get_neighborhood_feature(diagram, row, col):
    feature = [0] * 5  # For 5 possible colors
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= row + i < len(diagram) and 0 <= col + j < len(diagram[0]):
                color = diagram[row + i][col + j]
                encoded_color = one_hot_encode(color)
                feature = [f + e for f, e in zip(feature, encoded_color)]
    return feature

def get_color_transition_feature(diagram, row, col):
    transitions = 0
    current_color_encoded = one_hot_encode(diagram[row][col])
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= row + i < len(diagram) and 0 <= col + j < len(diagram[0]):
                neighbor_color = diagram[row + i][col + j]
                neighbor_color_encoded = one_hot_encode(neighbor_color)
                if current_color_encoded != neighbor_color_encoded:
                    transitions += 1
    return [transitions]

# Function to count colors in the diagram
def count_colors(diagram):
    color_count = {'R': 0, 'B': 0, 'Y': 0, 'G': 0, 'W': 0}  # Initialize count for each color
    for row in diagram:
        for cell in row:
            color_count[cell] += 1
    return list(color_count.values())

# Function to count color transitions in rows and columns
def count_color_transitions(diagram):
    row_transitions = [0] * len(diagram)
    col_transitions = [0] * len(diagram[0])

    # Count transitions in rows
    for i, row in enumerate(diagram):
        for j in range(1, len(row)):
            if row[j] != row[j-1]:
                row_transitions[i] += 1

    # Count transitions in columns
    for j in range(len(diagram[0])):
        for i in range(1, len(diagram)):
            if diagram[i][j] != diagram[i-1][j]:
                col_transitions[j] += 1

    return row_transitions + col_transitions

# Generate dataset
def create_dataset(num, task_number):
    data = []
    labels = []

    for _ in range(num):
        if task_number == 1:
            # Task 1: Predicting if a diagram is dangerous
            diagram, wire_order, arrayInput = create_diagram()
            feature_vector = count_colors(diagram)  # Add color counts to feature vector

            # Existing feature extraction
            for color in arrayInput:
                feature_vector.extend(one_hot_encode(color))
            for row_idx in range(len(diagram)):
                for col_idx in range(len(diagram[0])):
                    feature_vector.extend(get_neighborhood_feature(diagram, row_idx, col_idx))
                    #feature_vector.extend(get_color_transition_feature(diagram, row_idx, col_idx))

            label = 1 if "R" in wire_order and "Y" in wire_order and wire_order.index("R") < wire_order.index("Y") else 0

        elif task_number == 2:
            # Task 2: Predicting which wire to cut in a dangerous diagram
            diagram, wire_order, arrayInput, wire_to_cut = create_dangerous_diagram()
            feature_vector = []

            # Add color transitions to feature vector
            feature_vector.extend(count_color_transitions(diagram))

            # Existing feature extraction
            for color in arrayInput:
                feature_vector.extend(one_hot_encode(color))
            for row_idx in range(len(diagram)):
                for col_idx in range(len(diagram[0])):
                    feature_vector.extend(get_neighborhood_feature(diagram, row_idx, col_idx))
                    feature_vector.extend(get_color_transition_feature(diagram, row_idx, col_idx))

            wire_to_cut_index = ['R', 'B', 'Y', 'G'].index(wire_to_cut)
            label = [1 if i == wire_to_cut_index else 0 for i in range(4)]
            data.append(feature_vector)
            labels.append(label)

        # Add to dataset
        data.append(feature_vector)
        labels.append(label)

    return data, labels




def split_data(data, labels, train_ratio=0.9):
    combined = list(zip(data, labels))
    random.shuffle(combined) #do i need to shuffle this?
    train_size = int(len(combined) * train_ratio)
    train, test = combined[:train_size], combined[train_size:]
    return train, test



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



#IMPLEMENTATION

# Declare the task 
taskN = int(input("Enter the task number (1 or 2): "))
#create data
data, labels = create_dataset(2000, taskN)

#split dataset
train_data, test_data = split_data(data, labels)
x_train, y_train = list(zip(*train_data))  
x_test, y_test = list(zip(*test_data))    


# Apply standardization to your data
x_train_scaled = standardize(x_train)
x_test_scaled = standardize(x_test)

#run models
if taskN == 1:
    # Logistic Regression for Task 1
    model = LogisticRegressionModel(len(x_train_scaled[0]), reg_lambda=0.01)
    model.train(x_train_scaled, list(y_train), epochs=100, learning_rate=0.02)

    # Testing the model
    correct_predictions = 0
    for x, y_true in zip(x_test_scaled, y_test):
        y_pred = model.predict(x)
        correct_predictions += 1 if (y_pred > 0.5) == y_true else 0

    accuracy = correct_predictions / len(x_test_scaled) * 100

    # After training, calculate predictions for the test set
    y_pred_test = [model.predict(x) for x in x_test_scaled]

    # Calculate the average log loss on the test set
    average_log_loss = log_loss(y_test, y_pred_test)

    print(f"Model accuracy for Task 1: {accuracy}%")
    print(f"Average Log Loss for Task 1: {average_log_loss}")

elif taskN == 2:
    # Softmax Regression for Task 2
    model = SoftmaxRegressionModel(len(x_train_scaled[0]), 4, reg_lambda=0.01)
    model.train(x_train_scaled, list(y_train), epochs=100, learning_rate=0.01)

    # Evaluate the model
    accuracy = model.evaluate(x_test_scaled, y_test)

    # After training, calculate predictions for the test set
    y_pred_test = [model.predict(x) for x in x_test_scaled]

    # Calculate the average log loss on the test set for multi-class classification
    average_log_loss = multi_class_log_loss(y_test, y_pred_test)

    print(f"Model accuracy for Task 2: {accuracy}%")
    print(f"Average Log Loss for Task 2: {average_log_loss}")
