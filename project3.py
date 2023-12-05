import random

def create_diagram():
    # Define the colors with ANSI escape codes for terminal colors
    colors = ['Red', 'Blue', 'Yellow', 'Green']
    color_codes = {
        'Red': '\033[91m*\033[0m',  # Red
        'Blue': '\033[94m*\033[0m',  # Blue
        'Yellow': '\033[93m*\033[0m', # Yellow
        'Green': '\033[92m*\033[0m',  # Green
        'White': '\033[97m*\033[0m'   # White
    }
    single_letter_codes = {
        'Red': 'R',
        'Blue': 'B',
        'Yellow': 'Y',
        'Green': 'G',
        'White': 'W'
    }
    diagram = [[color_codes['White'] for _ in range(20)] for _ in range(20)]
    wire_order = []

    # Determine whether to start with a row or a column
    start_with_row = random.choice([True, False])

    for _ in range(4):
        chosen_color = random.choice(colors)
        colors.remove(chosen_color)
        wire_order.append(chosen_color)

        if start_with_row:
            # Color a row
            row = random.randint(0, 19)
            for i in range(20):
                diagram[row][i] = color_codes[chosen_color]
            start_with_row = False  # Switch to column for the next iteration
        else:
            # Color a column
            col = random.randint(0, 19)
            for i in range(20):
                diagram[i][col] = color_codes[chosen_color]
            start_with_row = True  # Switch to row for the next iteration

    flattened_diagram = []
    for row in diagram:
        for cell in row:
            for color, code in color_codes.items():
                if cell == code:
                    flattened_diagram.append(single_letter_codes[color])

    return diagram, wire_order, flattened_diagram

# Create and print the diagram and wire order
diagram, wire_order, flattened_diagram = create_diagram()
for row in diagram:
    print(' '.join(row))

print("\nOrder of wires laid down:", wire_order)

# Check if the image is 'Dangerous' or 'Safe'
status = "Dangerous" if "Red" in wire_order and "Yellow" in wire_order and wire_order.index("Red") < wire_order.index("Yellow") else "Safe"
print("Status:", status)

# Identify the wire to cut if dangerous
if status == "Dangerous":
    print("Wire to cut:", wire_order[2])

# Convert the diagram into a 1D array
print("\nFlattened Diagram:", flattened_diagram)
