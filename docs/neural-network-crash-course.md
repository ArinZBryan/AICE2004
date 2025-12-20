# Neural Network Crash Course
This should give you a brief overview of the computations involved in a neural network, so you have a better high-level idea of what's going on.

**NOTE**: This is not examined, or even really required to do the coursework,
but may be of interest in understanding what the programe is doing.

**Disclaimer**: This document can be changed to reflect any queries about neural networks.
If you're confused about anything please register an issue at https://git.soton.ac.uk/aice2004/coursework/-/issues.

## Problem Statement

When a neural network receives an input matrix, it goes through a set of algebraic transformations that should result in the classification you are interested in.

Let's take a sample from our Fashion-MNIST dataset to use as an example:

![Fashion MNIST example](img/fashion.png)

The picture shows a boot in 28×28 pixels (784 total); how could we transform this into the word 'boot'?

Each dataset comes with an accompanying set of labels that the dataset has examples of. For Fashion-MNIST it is `["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]`.

What we can do is simply *enumerate* that list *by its indices*. For example, "T-shirt/top" is 0, "Dress" is 3, and our "Ankle boot" would be 9.

Now, we have a simpler problem statement: "Turn an image, a matrix of 784 pixels, into a number that corresponds to an index in a corresponding label array".

In other words (or images):

![Problem visualization](img/prob.png)

You can see how a data sample looks like in code by examining the [DataLoader object](../include/DataLoader.h)

## Inference (Forward)

The `forward` method constitutes a set of linear shape matrix transformations that end up transforming the shape into what we expect, as specified in the problem statement.

During the forward pass (the inference), the network does the following with our input x:

1. **Slim down the input shape (784) to our hidden shape (100)**
   - x1: (100, 784) × (784) → (100)
   - x1 = weight[0] × input

2. **Add bias**
   - x2: (100) + (100) → (100)
   - x2 = x1 + bias[0]

3. **Apply the activation function, sigmoid (σ)**
   - x3: σ((100)) → (100)
   - x3 = σ(x2)

4. **Slim down the hidden shape (100) to the output shape, which corresponds to labels (10)**
   - x4: (10, 100) × (100) → (10)
   - x4 = weight[1] × x3

5. **Apply the bias to the output layer**
   - x5: (10) + (10) → (10)
   - x5 = x4 + bias[1]

6. **Apply softmax to get probabilities**
   - x6: softmax((10)) → (10)
   - x6 = softmax(x5)

You can examine this in code by looking at the [Network::forward](../src/Network.cpp) method.

## Backward

An untrained `forward` pass changes the structure to our expected one, but the resulting output means nothing if we do not correctly change the associated weights. This is what happens during the backward pass.

**Quick intuition:** The backward pass computes how much each weight contributed to the error in the output. We calculate gradients (derivatives) that tell us which direction to adjust each weight to reduce the error. By flowing these gradients backward through the network (from output to input), we can update all weights to make better predictions.

### Derivatives for Each Operation

Each operation in the forward pass has an associated derivative that tells us how to propagate the error backward:

| Operation             | Forward Function (f)  | Backward Gradient Flow                                    |
|-----------------------|-----------------------|-----------------------------------------------------------|
| Matrix multiplication | f(W, x) = Wx          | ∂L/∂W = ∂L/∂f · x<sup>T</sup>, ∂L/∂x = W<sup>T</sup> · ∂L/∂f |
| Bias addition         | f(x, b) = x + b       | ∂L/∂b = ∂L/∂f, ∂L/∂x = ∂L/∂f                            |
| Sigmoid (σ)           | f(x) = σ(x)           | ∂L/∂x = ∂L/∂f · σ(x)(1 - σ(x))                           |
| Softmax + Cross-Entropy | Combined              | ∂L/∂x = softmax(x) - target                              |

*Note: ∂L/∂f represents the gradient flowing backward from the next layer.*

### Backward Pass Steps

The backward pass starts from the output and applies the corresponding derivative to each operation, working backward through the network:

1. **Backward through cross-entropy loss**
   - This gives us the initial error gradient
   - z6: (10) - (10) → (10)
   - z6 = output - target

2. **Backward through bias addition (reverse of x5)**
   - The gradient for bias[1] is just z6
   - z5: (10)
   - z5 = z6
   - grad_bias[1] = z5

3. **Backward through weight multiplication (reverse of x4)**
   - Gradient for weight[1]: outer product of z5 and x3
   - Gradient flowing to x3: weight[1]<sup>T</sup> × z5
   - z4: (100)
   - z4 = weight[1]<sup>T</sup> × z5
   - grad_weight[1] = z5 × x3<sup>T</sup>

4. **Backward through sigmoid activation (reverse of x3)**
   - Apply sigmoid derivative: σ(x2) · (1 - σ(x2))
   - z3: (100)
   - z3 = z4 · σ'(x2) = z4 · x3 · (1 - x3)

5. **Backward through bias addition (reverse of x2)**
   - Gradient for bias[0] is just z3
   - z2: (100)
   - z2 = z3
   - grad_bias[0] = z2

6. **Backward through weight multiplication (reverse of x1)**
   - Gradient for weight[0]: outer product of z2 and input
   - z1 = weight[0]<sup>T</sup> × z2 (not needed, as input isn't trained)
   - grad_weight[0] = z2 × input<sup>T</sup>

### Updating Weights to Improve the Network

Now that we have gradients for all weights and biases, we update them using gradient descent:
```
weight[0] = weight[0] - learning_rate × grad_weight[0]
bias[0] = bias[0] - learning_rate × grad_bias[0]
weight[1] = weight[1] - learning_rate × grad_weight[1]
bias[1] = bias[1] - learning_rate × grad_bias[1]
```

The `learning_rate` controls how big of a step we take. By subtracting the gradient, we move the weights in the direction that *reduces* the error. 

**How does this train the network?** Each time we do a forward pass followed by a backward pass and weight update, the network's predictions become slightly more accurate. The gradients literally point the weights toward values that would have produced a better output for that input. After thousands of these updates across many training examples, the network learns to recognize patterns and make accurate predictions on new data.

You can examine this in code by looking at the [Network::backpropagate](../src/Network.cpp) method.
