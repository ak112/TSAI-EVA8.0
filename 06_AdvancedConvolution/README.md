## Advanced Convolution

1. change the code such that it uses GPU.
2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
   total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
    1. horizontal flip
    2. shiftScaleRotate
    3. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## Solution

A ConvNet architecture for the CIFAR-10 dataset classification task is designed. The network consists of several blocks of convolutional and activation layers.

   * The first block, "conv1", contains 4 convolutional layers (Conv2d), each followed by a ReLU activation layer, a batch            normalization layer (BatchNorm2d), and a dropout layer (Dropout) with a rate of 0.05. The convolutional layers have 128, 64, and 32 output channels and use 5x5, 3x3, and 3x3 kernels, respectively, with stride of 1 and padding of 2, 1, and 1, respectively.
   * The second block, "conv2", also contains several convolutional layers, with the first one using a dilation rate of 2. Between the convolutional layers, there are also activation and batch normalization layers as well as dropout layers.
   * The third block, "conv3", is similar to the second block, with a stride of 2 and dilation rate of 2 in the first layer.
   * The fourth block, "conv4", is similar to the second block, with a stride of 2 and dilation rate of 2 in the first layer.
   * The final block, "gap", is a global average pooling layer (AvgPool2d) with a kernel size of 3.
   * The output of the final block is reshaped into a vector with a size of 10 and passed through a log_softmax activation to get     the  final class scores for the 10 classes of the CIFAR-10 dataset.

For the purpose of augmenting the images we have used albumentations library. The network architecture created utilizes 159,552 params. The network was trained for 150 epochs and achieved the desired accuracy of 85%.
