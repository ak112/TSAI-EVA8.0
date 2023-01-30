## Objectives

1. You are making 3 versions of your 4th assignment's best model (or pick one from best assignments):  
    1. Network with Group Normalization  
    2. Network with Layer Normalization  
    3. Network with L1 + BN  
2. You MUST:  
    1. Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include  
    2. Write a single notebook file to run all the 3 models above for 20 epochs each  
    3. Create these graphs:
        - Graph 1: Test/Validation Loss for all 3 models together  
        - Graph 2: Test/Validation Accuracy for 3 models together  
        - graphs must have proper annotation  
    4. Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images.  
    5. write an explanatory README file that explains:  
        - what is your code all about,  
        - how to perform the 3 normalizations techniques that we covered(cannot use values from the excel sheet shared)
        - your findings for normalization techniques  
        - add all your graphs  
        - your 3 collection-of-misclassified-images  
    6. Upload your complete assignment on GitHub and share the link on LMS  
    
 ## Solution
The purpose of this code is to investigate the effect of different normalisation techniques and L1 regularisation on a CNN model trained on the MNIST dataset. It has the following characteristics:

a) To load the MNIST dataset, use Data Loaders.

b) A modularized model that allows you to choose the type of normalisation to use: batch normalisation (BN), group normalisation (GN), or layer normalisation (LN).

c) A function is defined for plotting misclassified images in order to better understand the model's performance in each of the three normalizations.

d) Plots to depict training and testing loss and accuracy during the training process.

Link to Notebook: https://colab.research.google.com/drive/1UKdapnaA_Q4zXWNjVVNHOJKnagVDsyAa
