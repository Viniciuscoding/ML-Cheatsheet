

# Basic Vectorization

## Challenges of text representation.
1. Convey some meaning.
2. Efficiency to feed into a ML model.

## One-hot Encoding

### Cons
1. No relationship between words.
2. Generates high-dimension and sparce vectors. This leads to overfitting.
### Pros
1. Intuitive to understand.
2. Easy to implement.

## Bag-of-words

### Pros
1. Intuitive do understand.
2. Easy to implement.
3. Captures some semantic similarity of texts.
4. Less sparce vectors than one-hot encoding.

### Cons
1. Generates high-dimension and sparse vectors.
2. Doesn't capture the relationship between words.
3. It does not consider the order of the words.

## Two Major Problems from bith models
1. High-dimension and sparce vectors.
2. Lack of relationship between words.


## Word Embeddings
It is a technic of distributive representation. In other words,
it captures the semantic meaning of words.

### Types of Word Embeddings:
1. Word2Vec (Google)
2. GloVe (Stanford)
3. FastText (Facebook)

### Pros
1. Low-dimension and dense vectors.
2. Captures the semantic meaning of words.
3. Vectors are trained by neural networks rather than manual work.

### Cons


### Word2Vec
1. It is not a singular algorithm.
2. It is a family of model architectures and optimizations.
that can be used to learn word embeddings from large datasets.
2.1 Model Architectures
2.1.2. Continous bag-of-words (CBOW)
Predicts the center word given the surrouding context words.
2.1.2. Skip-gram
Predicts the surrouding context words given the center word.

Input -> Embedding Matrix -> Hidden Layer -> Softmax -> Output Layer vs Actual Result

Input x Embedding Matrix = Embedded Vector

One-hot Encoding x word2vec = Embedded Vector
x(1*v) x E(v*d) = x(1*d)

Hidden Layer = SUM(Embedded Vector)
H(1*d) = x1(1*d) + x2(1*d) + ... xn(1*d)

Output Layer = Hidden Layer x Softmax Function
y(1*v) = H(1*d) x E'(d*v)

Actual Result = Y(1*v)

### CBOW calculation

### Skip-gram calculation


## Artificial Neural Network (ANN)

Activation Functions -> Cost (Loss) Fucntions -> Gradient Descent -> Backpropagation -> Learning Rate -> Epochs

### What is an **Activation Function** used for?
It is used to **prevent linearity**. It converts a linear network to a non-linear one.
### Types of Activation Functions
**Rectified Linear Unit (ReLU)**
`return if x < 0 then 0 else x`
**Sigmoid**
Transform all values between 0 and 1. Commonly used on binary-class classification and logistic regression models such as Email Spam detection.<br>
`f(x) = 1 / (1 + e^(-x))` OR return 1/(1+e**-x)
**Hyperbolic Tangent (tanh)**
Transform all values between -1 and 1<br>
`f(x) = (e^x - e^(-x))/e^x + e^(-x))`
**Softmax**
Commonly used on multi-class classification models such as costumer ratings.<br>
`fi(x) = e^x / ∑(J,j=1)e^(xj)) for i = 1,...,j`

## Cost functions or loss functions
Quantifies the comparison between predicted results versus actual results

### Cost (Loss) Function 
**Cost Function**: Used to compute errors of the entire training dataset. 
**Loss Function**: Used to compute errors of a single training dataset instance.
### Mean Squared Error (MSE)
Used for regression problems
`MSE = ∑(n,i=1)(Y'i - Yi)^2 / n`<br>
`Y'i = Predicted value | Yi = Actual value | n = Size of the training data`
### Cost functions for classification problems
### Cross-entropy
Calculates the differences between probability distributions.

### Backpropagation
Go back to adjust weights and paramaters in order to minimize the cost (loss) function.
How to adjust the weights?
### Gradient Descent
The process of walking down the surface formed to the process of walking down surface formed
by the cost function and finding the bottom.
### Finding the bottom
#### Which direction to take?
The derivative decides the direction to take
#### How large should the step be?
The learning rate (step size) determines the learning speed. It is a hyperparameter set before training.
1. If learning rate (step size) is too small it might take too long.
2. If learning rate (step size) is too large you my might not converge to the lowest point because 
it might bounce back and forth the same position or even outside the curve (overshoots).
### How many iterations to find the bottom?
How many iterations are needed for the cost function to hit the optimum (the bottom of its curve)?
The number of iterations is commonly called **epoch**. It is a hyperparameter that is set until values can't go down anymore.
