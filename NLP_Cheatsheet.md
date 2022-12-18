

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
