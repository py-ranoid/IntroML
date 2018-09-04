In this assignment you will practice writing backpropagation code, and training
Neural Networks and Convolutional Neural Networks. The goals of this assignment
are as follows:

- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **batch normalization** for training deep networks
- implement **dropout** to regularize networks
- effectively **cross-validate** and find the best hyperparameters for Neural
  Network architecture
- understand the architecture of **Convolutional Neural Networks** and train
  gain experience with training these models on data

### Working locally
Get the code as a zip file
[here](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment2.zip).
As for the dependencies:


**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `assignment2` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```

**Compile the Cython extension:** Convolutional Neural Networks require a very
efficient implementation. We have implemented of the functionality using
[Cython](http://cython.org/); you will need to compile the Cython extension
before you can run the code. From the `cs231n` directory, run the following
command:

```bash
python setup.py build_ext --inplace
```

**Start IPython:**
After you have the CIFAR-10 data, you should start the IPython notebook server
from the `assignment2` directory. If you are unfamiliar with IPython, you should 
read our [IPython tutorial](http://cs231n.github.io/ipython-tutorial/).

**NOTE:** If you are working in a virtual environment on OSX, you may encounter
errors with matplotlib due to the
[issues described here](http://matplotlib.org/faq/virtualenv_faq.html).
You can work around this issue by starting the IPython server using the
`start_ipython_osx.sh` script from the `assignment2` directory; the script
assumes that your virtual environment is named `.env`.

### Q1: Fully-connected Neural Network
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Q2: Batch Normalization
In the IPython notebook `BatchNormalization.ipynb` you will implement batch
normalization, and use it to train deep fully-connected networks.

### Q3: Dropout
The IPython notebook `Dropout.ipynb` will help you implement Dropout and explore
its effects on model generalization.

### Q4: ConvNet on CIFAR-10
In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several
new layers that are commonly used in convolutional networks. You will train a
(shallow) convolutional network on CIFAR-10, and it will then be up to you to
train the best network that you can.

### Q5: Do something extra!
In the process of training your network, you should feel free to implement
anything that you want to get better performance. You can modify the solver,
implement additional layers, use different types of regularization, use an
ensemble of models, or anything else that comes to mind. If you implement these
or other ideas not covered in the assignment then you will be awarded some bonus
points.

