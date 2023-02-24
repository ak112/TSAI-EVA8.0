## Problem Statement


Build the following network:
1. That takes a CIFAR10 image (32x32x3)
2. Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
3. Apply GAP and get 1x1x48, call this X
4. Create a block called ULTIMUS that:
    1. Creates 3 FC layers called K, Q and V such that:
        1. X*K = 48*48x8 > 8
        2. X*Q = 48*48x8 > 8 
        3. X*V = 48*48x8 > 8 
    2. then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
    3. then Z = V*AM = 8*8 > 8
    4. then another FC layer called Out that:
        1.Z*Out = 8*8x48 > 48
5. Repeat this Ultimus block 4 times
6. Then add final FC layer that converts 48 to 10 and sends it to the loss function.
7. Model would look like this C>C>C>U>U>U>U>FFC>Loss
8. Train the model for 24 epochs using the OCP that I wrote in class. Use ADAM as an optimizer. 
9. Submit the link and answer the questions on the assignment page:
    1. Share the link to the main repo (must have Assignment 7/8/9 model7/8/9.py files (or similarly named))
    2. Share the code of model9.py
    3. Copy and paste the Training Log
    4. Copy and paste the training and validation loss chart


## Solution

The network defined as a part of solution consists of several layers of convolutional and linear operations followed by four UltimusBlocks and a final fully connected layer.

The input images are 3-channel images, and the first three layers are convolutional layers with 16, 32, and 48 filters respectively, each with a kernel size of 3 and padding of 1. ReLU activation function is applied after each convolution operation.

After the third convolutional layer, an adaptive average pooling operation is performed on the feature map, which reduces the spatial dimensions to 1x1 while preserving the depth (number of channels).

The UltimusBlock module is a self-attention mechanism that takes a tensor of shape (batch_size, num_channels) as input and produces an output tensor of the same shape. The module is composed of three linear transformations, K, Q, and V, each with a dimensionality of 8. The dot product of Q and K is used to compute the attention weights, which are then used to weigh the values V to obtain the final output.

Four instances of UltimusBlock are stacked together in the architecture to capture increasingly complex relationships between the features. The final output of the last UltimusBlock is flattened and fed into a linear layer with 10 output units, which produces the logits for the 10-class classification problem.

