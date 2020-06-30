# Deeplearning.ai notes

## Popular notes

[mbadry1/DeepLearning.ai-Summary](https://github.com/mbadry1/DeepLearning.ai-Summary)

[ppant/deeplearning.ai-notes](https://github.com/ppant/deeplearning.ai-notes)

In general, the number of neurons in the previous layer gives us the number of columns of the weight matrix, and the number of neurons in the current layer gives us the number of rows in the weight matrix.

## Notation

![Deeplearning%20ai%20notes%20ef1397f591e74c4fb69de691d3284f74/Standard_notations.png](Deeplearning%20ai%20notes%20ef1397f591e74c4fb69de691d3284f74/Standard_notations.png)

# Week 1

## Bias and Variance

High variance —> Overfitted model

High bias —> Unability of model to truly capture the relationship.

A model with high bias is underfitted.

## L2 regularisation

$$$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}$$$

$$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}$$

**Observations**:

- The value of $\lambda$ is a hyperparameter that you can tune using a dev set.
- L2 regularization makes your decision boundary smoother. If $\lambda$ is too large, it is also possible to "oversmooth", resulting in a model with high bias.

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

**What you should remember** -- the implications of L2-regularization on:

- The cost computation:
    - A regularization term is added to the cost
- The backpropagation function:
    - There are extra terms in the gradients with respect to weight matrices
- Weights end up smaller ("weight decay"):
    - Weights are pushed to smaller values.

## **Dropout Regularization**

- In most cases Andrew Ng tells that he uses the L2 regularization.
- The dropout regularization eliminates some neurons/weights on each iteration based on a probability.
- A most common technique to implement dropout is called "Inverted dropout".
- Code for Inverted dropout:

    ```python
    keep_prob = 0.8 # 0 <= keep_prob <= 1
    l = 3 # this code is only for layer 3
    # the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
    d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob
    a3 = np.multiply(a3,d3) # keep only the values in d3
    # increase a3 to not reduce the expected value of output
    # (ensures that the expected value of a3 remains the same) - to solve the scaling problem
    a3 = a3 / keep_prob
    ```

- Vector d[l] is used for forward and back propagation and is the same for them, but it is different for each iteration (pass) or training example.
- At test time we don't use dropout. If you implement dropout at test time - it would add noise to predictions.
- How does dropout work?

    Dropout works by spreading the weights across different neurons since a certain set of neurons 
    are randomly switched off according the probability. It has a similar effect like l2 regularisation.

- Dropout is used a lot in computer vision application because a lot of times the data is less so chance of overfitting is high.

### Disadvantage

- The cost function is not well defined now since we are randomly subtracting a term.

What Andrew does to tackle the above problem?

- Turn off dropout, run code and make sure J is decreasing monotonically and then turn on hoping that no bugs crept.

## Other regularization methods

### Data augmentation

Flip, rotate , take random crops of images to create fake samples. These are not as good as 
good original images.

### Early stopping

Stop training when dev set error started increasing. BY stopping halfway, you have mid sized W, similar to l2 regularisation with a smaller wieght, the neural network does not ovefit.

So you look for the iteration where NN is doing it's best and stop training the neural network and take whatever value the dev set gives.

- Andrew prefers to use L2 regularization instead of early stopping because this technique simultaneously tries to minimize the cost function and not to overfit which contradicts the orthogonalization approach (will be discussed further).But its advantage is that you don't need to search a hyperparameter like in other regularization approaches (like `lambda` in L2 regularization).

## Speeding up training and setting up your optimization problem

### Normalizing inputs

- Make the mean 0 and variance 1 for the training set.

Why normalize?

- If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then optimizing it will take a long time.
- But if we normalize it the opposite will occur. The shape of the cost function will be consistent (look more symmetric like circle in 2D example) and we can use a larger learning rate alpha - the optimization will be faster.

### Vanishing/Exploding gradients and Solution 
by Weight Initialization

Setting initialization part inside sqrt to

2/n[l-1] for ReLU is better: $2/n[l-1]$

```
np.random.rand(shape) * np.sqrt(2/n[l-1])
```

**What you should remember from this notebook**:

Different initializations lead to different resultsRandom initialization is used to break symmetry and make sure different hidden units can learn different thingsDon't intialize to values that are too largeHe initialization works well for networks with ReLU activations.

Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)

---

# Week 2

## Mini-batch grad descent

If one has a small dataset —> Batch grad descent 

If one has really large dataset, mini batch comes into the picture.

Here, we divide the training set into various blocks of fixed sizes preferably 
in powers of 2(Helps as computer memory is binary) (32, 64, 128, 256, 512,1024) are good sizes.

- Batch gradient descent:
    - too long per iteration (epoch)
- Stochastic gradient descent:
    - too noisy regarding cost minimization (can be reduced by using smaller learning rate)
    - won't ever converge (reach the minimum cost)
    - lose speedup from vectorization
- Mini-batch gradient descent:
    1. faster learning:
        - you have the vectorization advantage
        - make progress without waiting to process the entire training set
    2. doesn't always exactly converge (oscelates in a very small region, but you can reduce learning rate)

Mini-batch size:

- (`mini batch size = m`) ==> Batch gradient descent
- (`mini batch size = 1`) ==> Stochastic gradient descent (SGD)
- (`mini batch size = between 1 and m`) ==> Mini-batch gradient descent

- Guidelines for choosing mini-batch size:
    1. If small training set (< 2000 examples) - use batch gradient descent.
    2. It has to be a power of 2 (because of the way computer memory is layed out and accessed, sometimes your code runs faster if your mini-batch size is a power of 2): `64, 128, 256, 512, 1024, ...`
    3. Make sure that mini-batch fits in CPU/GPU memory.
- Mini-batch size is a `hyperparameter`.

**Some doubts I had cleared by [Sahil](https://github.com/sahilkhose)**

- Epoch is the time needed for the whole dataset to go through one training pass
so whatever small batch you have epochs, will remain same.
- Suppose we have 10 mini-batches and 2 epochs, then total steps = 2 * 10
So, we will be having a higher number of gradient steps.

Loss function is evaluated faster.

```
for t = 1:No_of_batches                         # this is called an epoch
	AL, caches = forward_prop(X{t}, Y{t})
	cost = compute_cost(AL, Y{t})
	grads = backward_prop(AL, caches)
	update_parameters(grads)
```

### Exponential average

$$V(t) = \beta * v_{(t-1)} + (1-\beta) * \theta(t)$$

This equation is kind of beautiful. It is sort of a running weighted average.

- Gives more weightage to recent contributions
- Lesser contribution to older contributions as we proceed.

If we plot this it will represent averages over $-(1/(1-\beta)$

entries:

- `beta = 0.9` will average last 10 entries
- `beta = 0.98` will average last 50 entries
- `beta = 0.5` will average last 2 entries

### Bias correction

- Because `v(0) = 0`, the bias of the weighted averages is shifted and the accuracy suffers at the start.
- To solve the bias issue we have to use this equation:

    $$v(t) = (\beta * v_{(t-1)} + (1-\beta) * \theta(t)) / (1 - \beta^t)$$

### Grad descent with momentum

We use the idea of exponential average here. This reduces the vertical oscillations' mean and
makes the horizontal learning faster. (Check video by Andrew Ng)

```
vdW = 0, vdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dw, db on current mini-batch                
			
	vdW = beta * vdW + (1 - beta) * dW
	vdb = beta * vdb + (1 - beta) * db
	W = W - learning_rate * vdW
	b = b - learning_rate * vdb
```

People don't implement bias correction often.

$**\beta = 0.9$   tends to work well. Beta is also a hyperparameter.**

Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp (described in lecture) and Momentum.

**How does Adam work?**

1. It calculates an exponentially weighted average of past gradients, and stores it in variables $v$ (before bias correction) and $v^{corrected}$ (with bias correction).
2. It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables $s$ (before bias correction) and $s^{corrected}$ (with bias correction).
3. It updates parameters in a direction based on combining information from "1" and "2".

The update rule is, for   $l = 1, ..., L$:

$\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
v^{corrected}{dW^{[l]}} = \frac{v{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
s^{corrected}{dW^{[l]}} = \frac{s{dW^{[l]}}}{1 - (\beta_2)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}{dW^{[l]}}}{\sqrt{s^{corrected}{dW^{[l]}}} + \varepsilon}
\end{cases}
where:$

- t counts the number of steps taken of Adam
- L is the number of layers
- $\beta_1$ and $\beta_2$ are hyperparameters that control the two exponentially weighted averages.
- $\alpha$ is the learning rate
- $\varepsilon$ is a very small number to avoid dividing by zero

As usual, we will store all parameters in the `parameters` dictionary

# Week 3

### Hyper-Parameter Tuning

Tuning is required to obtain the best out of the model.

- We need to tune our hyperparameters to get the best out of them.
- Hyperparameters importance are (as for Andrew Ng):
    1. Learning rate.
    2. Momentum beta.
    3. Mini-batch size.
    4. No. of hidden units.
    5. No. of layers.
    6. Learning rate decay.
    7. Regularization lambda.
    8. Activation functions.
    9. Adam `beta1`, `beta2` & `epsilon`.

### Grid

Make a grid with N hyperparameter setting and then try out all combinations.
Andrew Ng suggests to try random values from the grid. This will save a lot of time and resources.

### Coarse to fine

When you find an area where your model works good, zoom into that area and try out more values. Try surrounding area values also. 

### Using an appropriate scale to search for hyper-parameters

Andrew suggests to use a logarithmic scale for hyper-parameters like beta (momentum) since the 
value of beta has a huge impact when it is near 1.

Beta's best range is from [0.9,0.999]. So, we need to search in 1 - [0.001,0.1]

```
a_log = np.log(a) # e.g. a = 0.001 then a_log = -3
b_log = np.log(b) # e.g. b = 0.1 then b_log = -1

r = (a_log - b_log) * np.random.rand() + b_log
# In the example the range would be from [-4, 0] because rand range [0,1)
beta = 1 - 10^r   # because 1 - beta = 10^r

```

## Pandas vs. Caviar approach

Pandas have only a single offspring(it seems) so they have to give utmost care to their child.
If you have less computational resources, then you may

- Start a model training with random parameters
- When you see the cost function/learning curve decrease gradually, you "nudge" the parameters a bit
- You can adjust the parameters as it trains.
- Panda approach

The fish lay millions of eggs in a year. They have to just make sure that some of their offspring survives

If you have enough computational resources,

- Train multiple models in parallel, check results. Repeat.

### Batch Normalization

- Proposed by Sergey Ioffe and Christian Szegedy.

Why use? It speeds up the learning process.

We normalize the inputs for faster convergence and better learning. Using the same idea,
the activations are the inputs for the next layers. So, we normalize the activations.

Andrew suggests that normalize Z[L-1] instead of the activations A[l-1]. There is some debate on this in the DL community.

![Deeplearning%20ai%20notes%20ef1397f591e74c4fb69de691d3284f74/Batch_Normalization_-_EXPLAINED!_6-13_screenshot.png](Deeplearning%20ai%20notes%20ef1397f591e74c4fb69de691d3284f74/Batch_Normalization_-_EXPLAINED!_6-13_screenshot.png)

**The gamma and beta are learnable parameters. They are learnt by the optimization algorithm.**
Thus, the mean and variance are not necessarily 0 and 1. The parameters
**allow to restore the identity mapping.**

Shapes:

- `Z[l] - (n[l], m)`
- `beta[l] - (n[l], m)`
- `gamma[l] - (n[l], m)`

### Why batch normalization works

1. Same reason as normalizing inputs
2. It makes the optimization landscape much smoother
3. Helps to deal with the problem of [covariate shift](https://qr.ae/pNKec1) (shifting of the input distribution away from normalization due to successive transformations i.e when the neural network learns and weights are updated, the distribution of outputs of a specific layer in the network changes. This forces the higher layers to adapt to that drift, which slows down learning.)
4. It has some regularization effect as well. But don't use batch normalization for regularization.

### **Batch norm at test time**

We test only single samples (mostly) during testing time. So, we are not training entire mini-batches. Hence we need to maintain a weighted average for gamma, beta.
This is known as the running average 

### Softmax Classifier

The Softmax classifier uses the cross-entropy loss. The Softmax classifier gets its name from the softmax function, which is used to squash the raw class scores into normalized positive values that sum to one, so that the cross-entropy loss can be applied. 

The idea is to express the chance in positive values. So, in the numerator we have
the e^(Z[L]) and denominator has the summation of all classes(sum = 1). Hence, we have a probability for all the classes. 

Softmax activation equations:

```
t = e^(Z[L])                      # shape(C, m)
A[L] = e^(Z[L]) / sum(t)          # shape(C, m), sum(t) - sum of t's for each example (shape (1, m))
```

C is the no. of classes

The loss function used with softmax:

```
L(y, y_hat) = - sum(y[j] * log(y_hat[j])) # j = 0 to C-1
```

### Deep learning frameworks

Popular ones(my list)

- Tensorflow
- Pytorch
- Fastai
- Keras

### Tensorflow

This was my first exposure to tensorflow. The static graph felt a bit annoying as I am more used 
to Pytorch with eager mode execution and dynamic computation graph. 

Writing and running programs in TensorFlow has the following steps:

1. Create Tensors (variables) that are not yet executed/evaluated.
2. Write operations between those Tensors.
3. Initialize your Tensors.
4. Create a Session.
5. Run the Session. This will run the operations you'd written above.
