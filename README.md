# AI-Project-3
## Overview
This project implements a machine learning model to analyze wiring diagrams and make critical decisions based on the diagram's layout. The primary focus is assessing whether a wiring diagram is 'Safe' or 'Dangerous' and determining which wire should be cut in the case of dangerous diagrams.

## Capabilities
**Safety Classification:** The model classifies each wiring diagram as either 'Safe' or 'Dangerous', based on the configuration of the wires. This classification is done with a custom logistic regression model developed from scratch.

**Wire Selection for Cutting:** In scenarios where the wiring is deemed dangerous, the model identifies the specific wire that needs to be cut. This decision is based on the rule that the third wire laid down is the one to be cut.

**Adaptability and Expansion:** While currently tailored to a specific rule set, the underlying framework of the model allows for adaptability and expansion. It can be adjusted to accommodate different rules or more complex decision-making criteria.

## How It Works
**Data Preparation:** Wiring diagrams are converted into a structured numerical format suitable for machine learning. Each diagram is represented as a vector of features, capturing essential aspects of the diagram.

**Logistic Regression Model:** A logistic regression model, built from the ground up, is used to classify diagrams. It utilizes gradient descent for optimization, minimizes a custom log loss function, and uses regularization to avoid overfitting.

**Decision Making for Dangerous Diagrams:** When a diagram is classified as dangerous, the softmax model applies multi-class classification to determine which wire to cut.

**Testing and Validation:** The model is thoroughly tested and validated against a diverse set of wiring diagrams to ensure accuracy and reliability in its predictions.
