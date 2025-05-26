 import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # weights - keeping it simple
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias
        
        # cache for backprop
        self.h = None
        self.y = None
        
    def forward(self, inputs, h0=None):
        # inputs shape: (seq_len, input_size)
        # h0: initial hidden state (optional)
        
        seq_len = inputs.shape[0]
        if h0 is None:
            h0 = np.zeros((self.Whh.shape[0], 1))
            
        # store hidden states for backprop
        self.h = [h0]
        self.y = []
        
        # forward pass - just matrix mult and tanh
        for t in range(seq_len):
            # hidden state
            h_next = np.tanh(
                np.dot(self.Wxh, inputs[t:t+1].T) + 
                np.dot(self.Whh, self.h[-1]) + 
                self.bh
            )
            self.h.append(h_next)
            
            # output
            y = np.dot(self.Why, h_next) + self.by
            self.y.append(y)
            
        return np.array(self.y)
    
    def backward(self, inputs, targets, learning_rate=0.01):
        # basic BPTT -- not the most efficient but works
        
        seq_len = len(targets)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dhnext = np.zeros_like(self.h[0])
        
        # backward pass
        for t in reversed(range(seq_len)):
            # output gradients
            dy = self.y[t] - targets[t]
            dWhy += np.dot(dy, self.h[t+1].T)
            dby += dy
            
            # hidden gradients
            dh = np.dot(self.Why.T, dy) + dhnext
            
            # backprop through tanh
            dh_raw = (1 - self.h[t+1] * self.h[t+1]) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t:t+1])
            dWhh += np.dot(dh_raw, self.h[t].T)
            
            # next hidden state
            dhnext = np.dot(self.Whh.T, dh_raw)
            
        # update weights - simple SGD
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

# quick test
if __name__ == "__main__":
    # tiny example - predict next number in sequence
    rnn = VanillaRNN(input_size=1, hidden_size=4, output_size=1)
    
    # simple sequence: 1,2,3,4 -> predict 5
    X = np.array([[1], [2], [3], [4]])
    y = np.array([[5]])
    
    # train for a few steps
    for _ in range(100):
        out = rnn.forward(X)
        rnn.backward(X, y)
        
    print("prediction:", rnn.forward(X)[-1][0])  # should be close to 5