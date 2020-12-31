# pytorch_projects
Implementation of classic models in pytorch. Prior to executing any file, please check whether the python modules have already been installed. 
- MNIST: classification of hand-written digits
    - Type 1: if you have downloaded the MNIST data from [this site](http://yann.lecun.com/exdb/mnist/)
    - Type 2: the data should be downloaded by pytorch automatically. 
    - Accuracy over 97.5% after 10 epochs. 
- Seq2seq: sequence to sequence machine translation
    - Data is loaded with torchtext.dataset, available on [this site](https://github.com/multi30k/dataset)
    - Obtain low perplexity (one-digit number) and high accuracy after around 10 epochs. 
    - Feel free to choose different combinations of source and target languages. 
    - There are two implementations of the same model:
        - seq2seq.py: the code for forward pass and accuracy presented. 
        - seq2seq.py: rely more on torch functions. 