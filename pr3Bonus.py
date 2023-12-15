import tensorflow as tf
import numpy as np
import random
import math

# THIS IS FOR TASK1
# Function to create a non-dangerous wire diagram for Task 1
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
# Function to create a dangerous wire diagram for Task 2
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


# Function to count colors in a neighborhood for a given cell
def get_neighborhood_color_counts(diagram, row, col):
    color_count = {'R': 0, 'B': 0, 'Y': 0, 'G': 0, 'W': 0}
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= row + i < len(diagram) and 0 <= col + j < len(diagram[0]):
                color = diagram[row + i][col + j]
                color_count[color] += 1
    return list(color_count.values())
    
# Function to count local color transitions for a given cell
def get_local_color_transitions(diagram, row, col):
    transitions = 0
    current_color = diagram[row][col]
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= row + i < len(diagram) and 0 <= col + j < len(diagram[0]):
                neighbor_color = diagram[row + i][col + j]
                if current_color != neighbor_color:
                    transitions += 1
    return transitions

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

# Data Preparation Functions 
def create_dataset(num, task_number):
    data = []
    labels = []

    for _ in range(num):
        if task_number == 1:
            # Task 1: Predicting if a diagram is dangerous
            diagram, wire_order, _ = create_diagram()
            feature_vector = np.array(count_colors(diagram))  # Global color counts

            # Add local neighborhood features
            local_features = np.array([get_neighborhood_color_counts(diagram, row_idx, col_idx) 
                                      for row_idx in range(len(diagram)) 
                                      for col_idx in range(len(diagram[0]))])
            feature_vector = np.concatenate((feature_vector, local_features.flatten()))

            label = 1 if "R" in wire_order and "Y" in wire_order and wire_order.index("R") < wire_order.index("Y") else 0

        elif task_number == 2:
            # Task 2: Predicting which wire to cut in a dangerous diagram
            diagram, _, _, wire_to_cut = create_dangerous_diagram()
            feature_vector = np.array(count_colors(diagram))  # Global color counts

            # Add local color transition features
            local_features = np.array([get_local_color_transitions(diagram, row_idx, col_idx) 
                                      for row_idx in range(len(diagram)) 
                                      for col_idx in range(len(diagram[0]))])
            feature_vector = np.concatenate((feature_vector, local_features.flatten()))

            wire_to_cut_index = ['R', 'B', 'Y', 'G'].index(wire_to_cut)
            label = tf.one_hot(wire_to_cut_index, 4).numpy()

        # Add the processed sample to the dataset
        data.append(feature_vector)
        labels.append(label)

    return np.array(data), np.array(labels)

def split_data(data, labels, train_ratio=0.9):
    dataset_size = len(data)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_size = int(dataset_size * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    x_train, y_train = data[train_indices], labels[train_indices]
    x_test, y_test = data[test_indices], labels[test_indices]

    return x_train, x_test, y_train, y_test

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-7)  # Adding a small value to avoid division by zero

# TensorFlow Model Definitions
def create_logistic_model(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(input_size,), activation='sigmoid')
    ])
    return model

def create_softmax_model(input_size, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, input_shape=(input_size,), activation='softmax')
    ])
    return model

# Main Execution Logic
def main(task_number):
    # Data generation and preprocessing
    data, labels = create_dataset(2000, task_number)
    x_train, x_test, y_train, y_test = split_data(data, labels)
    
    # Convert to TensorFlow tensors
    x_train_scaled = tf.convert_to_tensor(standardize(x_train), dtype=tf.float32)
    x_test_scaled = tf.convert_to_tensor(standardize(x_test), dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Model selection based on task
    if task_number == 1:
        model = create_logistic_model(len(x_train_scaled[0]))
        loss = 'binary_crossentropy'
    elif task_number == 2:
        model = create_softmax_model(len(x_train_scaled[0]), 4)
        loss = 'categorical_crossentropy'

    # Compile the model
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    # Train the model
    model.fit(x_train_scaled, y_train, epochs=100, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test_scaled, y_test)
    print(f"Accuracy: {accuracy}")

    # Additional code for predictions, saving model, etc. can be added here

if __name__ == "__main__":
    taskN = int(input("Enter the task number (1 or 2): "))
    main(taskN)
