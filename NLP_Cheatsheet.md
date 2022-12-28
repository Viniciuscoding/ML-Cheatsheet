# NLP Models

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
that can be used to learn word embeddings from large datasets.<br>
2.1 Model Architectures<br>
--2.1.2. Continous bag-of-words (CBOW)<br>
Predicts the center word given the surrouding context words.<br>
--2.1.2. Skip-gram<br>
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
#### **Rectified Linear Unit (ReLU)**
`return if x < 0 then 0 else x`<br>
#### **Sigmoid**
Transform all values between 0 and 1. Commonly used on binary-class classification and logistic regression models such as Email Spam detection.<br>
`f(x) = 1 / (1 + e^(-x))` OR return 1/(1+e**-x)<br>
#### **Hyperbolic Tangent (tanh)**
Transform all values between -1 and 1<br>
`f(x) = (e^x - e^(-x))/e^x + e^(-x))`<br>
#### **Softmax**
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

# Deep Neural Networks (DNN)
It is a neural network with multiple hidden layers.


# Recursive Neural Networks (RNN)

## PROS
1. It has a memory.
## CONS
1. It has a short-term memory.
## Vanishing Gradients
It is when the gradients of the paraementers with respect to the loss function are so small that it becomes difficult for the netwrok to learn.
### Solution?
It can be mitigated by using non-saturation behavior activation function such as **ReLu**, **Batch Normalization**, and **Residual Connections**.<br>
Or, using **Long Short-Term Memory (LSTM)**

# Long Short-Term Memory (LSTM)
It is a neural network with long-term memory.

## RNN vs LSTM
### RNN
1. One pipeline.
1.1. Shot-term memory pipeline.

### LSTM
1. Two pipelines.<br>
1.1. Short-term memory pipeline = hidden layer. Output layer is called hidden state.<br>
1.2. Long-term memory pipeline = cell layer. Output layer is called cell state.<br>
2. It has **Gates**.<br>
2.1. A Gate has a **Pointwise multiplication functions** and a **Sigmoid function**.<br>
2.2. In a Sigmoid function: 0 = forget information & 1 = remember information.<br>

NOTE: Gates in LSTM are divided in three sections:<br>
Section 1: **Forget Gate** decides what irrelevant information to forget.<br>
Section 2: **Input Gate** decides what new information should be remembered.<br>
Section 3: **Output Gate** decides what information should pass to the next time step.<br>

# Gated Recurrent Unit (GRU)
It is similar to LSTM but the mojor difference is that GRU merges the pipeline cell state and hidden state. In other words is like merging the short-term memory with the long-term memory.

## GRU Gates
It is an improved and simplified variant of LSTM.<br>
GRU has two major gates: **Reset Gate** and **Update Gate**.
### Reset Gate
It determines how much information from the past should be carried on to the future.
### Update Gate
It combines the forget and input gates in an LSTM into a single "update gate" to decide how much information should be forgotten and remembered.

# Convolution Neural Network (CNN)
It is widely used in computer vision tasks such as image processing, classification, and segmentation.<br>
Produces good results specially for relatively short-length sentences.



# Advanced NLP Models

## Encoder-Decoder

### Encoder
It is a network that can be built using unrolled recurrent neural network such as RNN, LSTM and GRU.

### Decoder


# Attention


# Transformer

## Multi-Headed Attention

### Encoders
Identical in structures: 6 Enconders<br>
Different in weights

### Encoder 2 Self-Layers
Self Attention<br>
Feedforward Neural Network

### Decoders

### Decoder 3 Self-Layers
Self Attention<br>
Encoder-Decoder Attention<br>
-- Focus on relevant parts of the input sentence
Feedforward Neural Network

Input -> Positional Encoding -> Embedding -> Query Vector | Key Vector | Value Vector<br> -> Self Attention Output -> Feedforward

Layers = 6<br>
Embedding Dimensionality = 512<br>
Econder Input/Output Dimensionality = 512<br>
Query Dimensionality = 64<br>
Key Dimensionality = 64<br>
Value Dimensionality = 64<br>
Attention Heads = 8

I + t -> x -> Q | K | V

Q | K | V = x * Wq | x * Wk | x * Wv<br>
Wq | Wk | Wv = Learned Query Weight | Learned Key Weight | Learned Value Weight

Embedding x Learned Query Weight = Query Vector<br>
Embedding x Learned Key Weight = Key Vector<br>
Embedding x Learned Value Weight = Value Vector<br>


softmax * (Q * K^t) / sqrt(dk)) * V = Z

Q = Query vector
K = Key vector
Kt = Transposed K
V = Value vector
dk = Dimensionality of K
Z = Self-attention output

Self-attention output -> Feeforward Neural Network<br>
Z -> F

Z x F

## Bidirectional Encoder Representations from Transformers (BERT)
0. Created by Google in 2018
1. Trained in two variations<br>
1.1. BERT Base: 12 Transformers Stacks -> 110 million parameters<br>
1.2. BERT Large: 24 Tranformers Stacks -> 340 million paramenters
2. Able to handle long input context
3. Trained on entire Wikipedia and BookCorpus
4. Trained for 1 million steps
5. Targeted at multi-task obejectives<br>
5.1. This is because it was trained on two variations: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
6. Trained on TPU for 4 days
7. Works at both sentence-level and token-level tasks
8. Can be fine-tuned for many different tasks


                     |BERT Base |BERT Large|Transformer|
|Layers              |    12    |    24    |     6     |
|FeedForward Networks|    768   |   1024   |    512    |
|Attention Heads     |    12    |    16    |     8     |

## BERT Input Embeddings
1. Token Embedding
2. Segment Embeddings
3. Position Embeddings

## BERT Usage
1. Single Sentence Classification
2. Sentence Pair Classification
3. Question Answering
4. Single Sentence Tagging Tasks


# Large Language Models

1. A single model can be used for different tasks
2. The fine-tune process require minimal filed data<br>
2.1. `few-shot` refers to training a model with minimal data<br>
2.2. `zero-shot` implies that a model can recognize things that have not explicitly been taught<br>
3. The performance is continuosly growing wiht more data and parameters





