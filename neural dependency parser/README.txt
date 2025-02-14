Neural Transition-Based Dependency Parser using PyTorch

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between
head words, and words which modify those heads. There are multiple types of dependency parsers,
including transition-based parsers, graph-based parsers, and feature-based parsers. This implementation
is a transition-based parser, which incrementally builds up a parse one step at a time. At every step
it maintains a partial parse, which is represented as follows:
• A stack of words that are currently being processed.
• A buffer of words yet to be processed.
• A list of dependencies predicted by the parser.
Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words
of the sentence in order. At each step, the parser applies a transition to the partial parse until its buffer
is empty and the stack size is 1. The following transitions can be applied:
• SHIFT: removes the first word from the buffer and pushes it onto the stack.
• LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of
the first item and removes the second item from the stack, adding a first word → second word
dependency to the dependency list.
• RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second
item and removes the first item from the stack, adding a second word → first word dependency to
the dependency list.

On each step, the parser decides among the three transitions using a neural network classifier.

Algorithm 1 Minibatch Dependency Parsing
Input: sentences, a list of sentences to be parsed and model, our model that makes parse decisions
Initialize partial parses as a list of PartialParses, one for each sentence in sentences
Initialize unfinished parses as a shallow copy of partial parses
while unfinished parses is not empty do
Take the first batch size parses in unfinished parses as a minibatch
Use the model to predict the next transition for each partial parse in the minibatch
Perform a parse step on each partial parse in the minibatch with its predicted transition
Remove the completed (empty buffer and stack of size 1) parses from unfinished parses
end while
Return: The dependencies for each (now completed) parse in partial parses.

The feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.). They can be represented as a list of integers w = [w1,w2,...,wm] where m is the number of features and each 0 ≤ wi < |V | is the index of a token in the vocabulary (|V | is the vocabulary size). Then our network looks up an embedding for each word and concatenates them into a single input vector:
x = [Ew1,...,Ewm] ∈ Rdm where E ∈ R|V |×d is an embedding matrix with each row Ew as the vector for a particular word w with dimension d. We then compute our prediction as:

h = ReLU(xW + b1) l = hU + b2
yˆ = softmax(l)

where h is referred to as the hidden layer, l is referred to as the logits, yˆ is referred to as the predictions, and ReLU(z) = max(z, 0)). We will train the model to minimize cross-entropy loss: J(θ) = CE(y,yˆ) = −Xyj logyˆj j=1 where yj denotes the jth element of y. To compute the loss for the training set, we average this J(θ) across all training examples.
