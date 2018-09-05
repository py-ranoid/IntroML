# The Math behind (almost) every AI model ever 
*So you think you can differentiate*
## Building your own Neural Network
Refer [this notebook by Andrew NG](https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb)

### What is a Neural Network
- An Input layer and an Output Layer
- One or more hidden layers in between to learn advanced relations
- Inspired by functioning of [Human Brain (and nervous system)](https://www.youtube.com/watch?v=0J5tM9bSbwA&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9&index=2)
- Intuition is that every relation (between any input and output) can be represented with mathematical function
- Hence, to predict output, find the best function (a.k.a weights)

### How do they work
- They're like linear regression on steriods
- *What if you used a linear regressor to predict inputs to another linear regressor** and so on
- Imagine cascaded/repeated linear regressors

![](https://i.kym-cdn.com/photos/images/newsfeed/000/531/557/a88.jpg)

- **How ?**
  1. Guess random function initially
  2. Predict output with function (Forward Propogation)
  3. Find loss (a.k.a cost)
  4. Update function to reduce loss next time (Back Propogation)
  5. Go to 2  
  
#### Forward propogation
- Every hidden has its own weights (Every hidden layer is like a function in itself)
  - Take input from previous layer
  - Multiply by its weights (*Do ML Stuff*)
  - Propogate activations (result) to next layer
- Repeat this for every layer between input and output.
- `1/2` ML done
- Forward because `input --> HiddenLayer1 --> HiddenLayer2 --> .... --> Output`
- Refer section 4.3 - Cell 11 in Notebook
#### BackPropogation
- Every layer's weights need to be updated (Every function needs to be improved)
  - Calculate Difference between predicted output and actual output (a.k.a Loss)
  - Use Loss to update weight in the direction where loss reduces (a.k.a [Gradient Descent](https://www.youtube.com/watch?v=nhqo0u1a6fw))
    - Calculate derivative of layer out wrt. layer weights
  - Update weights
  - Use this information to update previous layer's weights and so on
- Back/Backward because `input <-- HiddenLayer1 <-- HiddenLayer2 <-- .... <-- Output`
- Refer section 4.3 - Cell 12 & 13 in Notebook

#### Other details
- Epochs : How many times would you repeat ForwardPropogation + BackPropogation ?
- Stochastic Gradient Descent
  - Convention : Learn (FP+BP) from `nk` at once, in one epoch
  - Smarter & Faster to learn (FP+BP) from `k` samples `n` times in one epoch
  - How ?
    1. Shuffle dataset ("stochastic") ie. shuffle order of `nk` samples
    2. Iterate through dataset sequentially and pick `k` samples
    3. Ideally, this should "represent" the entire dataset
    4. FP & BP with `k` samples. Update weights.
    5. Process next `k` samples. ie. go to step 2.
  - Hence, within an epoch, we are learning `n` times
  - `k` is also called "Batch Size"
  
