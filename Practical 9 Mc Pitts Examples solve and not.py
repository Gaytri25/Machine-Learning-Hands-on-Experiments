
# ANDNOT Function Using McCulloch-Pitts Neural Network

def mp_neuron(x1, x2):

    # Step 2: Assign weights to inputs
    w1 = 1
    w2 = -1

    # Step 3: Set the threshold value
    threshold = 1

    # Step 4: Calculate the weighted sum
    net = (x1 * w1) + (x2 * w2)

    # Step 5: Apply the activation function
    if net >= threshold:
        output = 1
    else:
        output = 0

    # Return the final output
    return output


# Step 6: Display the heading
print("ANDNOT Function Using M-P Neuron")
print("--------------------------------")
print("X1\tX2\tNet\tOutput")
print("--------------------------------")

# Step 7: Test all possible input combinations
for x1 in [0, 1]:
    for x2 in [0, 1]:

        # Calculate weighted sum
        net = (x1 * 1) + (x2 * -1)

        # Find the output of the neuron
        output = mp_neuron(x1, x2)

        # Display the results
        print(x1, "\t", x2, "\t", net, "\t", output)

# Step 8: Display the truth table result
print("--------------------------------")
print("ANDNOT function executed successfully.")


