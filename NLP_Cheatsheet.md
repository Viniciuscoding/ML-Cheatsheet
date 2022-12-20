

# Basic Vectorization

## Challenges of text representation.
1. Convey some meaning.
2. Efficiency to feed into a ML model.

## One-hot Encoding

### Cons
1. No relationship between words
2. Generates high-dimension and sparce vectors. This leads to overfitting.
### Pros
1. Intuitive to understand
2. Easy to implement

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
2. It is a family of model architectures and optimizations
that can be used to learn word embeddings from large datasets.
2.1 Model Architectures
2.1.2. Continous bag-of-words (CBOW)
Predicts the center word given the surrouding context words 
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

### What is an **Activation Function** used for?
It is used to **prevent linearity**. It converts a linear network to a non-linear one.
### Types of Activation Functions
**Rectified Linear Unit (ReLU)**

