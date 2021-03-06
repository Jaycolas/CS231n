import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #h1: N*H
    h1 = X.dot(W1) + b1
    #For ReLu
    h1reLu = np.maximum(0,h1)
    #h2: N*C
    scores = h1reLu.dot(W2) + b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    scoresC = scores - np.max(scores)
    prob = np.exp(scoresC[range(N),y])/np.sum(np.exp(scoresC), axis=1)
    L = -np.log(prob)
    loss = np.sum(L)/N + 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    #loss = np.sum(-np.log(np.exp(scores[range(N),y])/np.sum(np.exp(scores), axis=1)),axis=1)/N + 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    #starts from loss to scores, W1, W2
    dloss_dW1 = reg*W1 #(D,H)
    dloss_dW2 = reg*W2 #(H,C)
    dloss_db1 = 0
    dloss_db2 = 0
    dloss_dscores = np.exp(scoresC)/np.sum(np.exp(scoresC),axis=1, keepdims=True)
    dloss_dscores[range(N), y]-=1  #(N,C)
    dloss_dscores/=N
    
    
    #Now starts from scores, W1,W2,b1,b2, h1_reLu
    dscores_dW1 = 0 # (D,H)
    dscores_dW2 = h1reLu # (N,H)
    dloss_scores_dW2 = np.dot(dscores_dW2.T, dloss_dscores) #(H,C)
    
    dscores_db1 = 0
    dscores_db2 = np.ones((1,N))
    dloss_scores_db2 = np.dot(dscores_db2,dloss_dscores) #(1,C)
    
    dscores_dh1reLu = W2 #(H,C)
    dloss_scores_dh1reLu = np.dot(dloss_dscores, dscores_dh1reLu.T)# (N,H)
    
    #Now starts from h1_ReLu
    dh1reLu_dh1 = np.zeros_like(h1reLu) #(N,H)
    dh1reLu_dh1[h1reLu==0]=0
    dh1reLu_dh1[h1reLu>0]=1
    dloss_scores_h1reLu_dh1 = dloss_scores_dh1reLu * dh1reLu_dh1 #(N,H)
    
    #Now starts from h1
    dh1_db1 = np.ones((1,N))
    dloss_scores_h1reLu_h1_db1 = np.dot(dh1_db1,dloss_scores_h1reLu_dh1) #(1,H)
    dh1_dW1 = X #(N,D)
    dloss_scores_h1reLu_h1_dW1 = np.dot(dh1_dW1.T, dloss_scores_h1reLu_dh1) #(D,H)
    dh1_db2 = 0
    dh1_dW2 = 0
    
    #Now calculate the total grad
    #(D,H)
    #X.T(D,N), (N,H), 
    dW1 = dloss_dW1 + dloss_scores_h1reLu_h1_dW1
    dW2 = dloss_dW2 + dloss_scores_dW2
    db1 = dloss_scores_h1reLu_h1_db1
    db2 = dloss_scores_db2
    grads['W1']=dW1
    grads['W2']=dW2
    grads['b1']=db1
    grads['b2']=db2
    
  
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      if num_train <= batch_size:
        X_batch = X
        y_batch = y
      else:
        mask = np.random.choice(num_train,batch_size,replace = True)
        X_batch = X[mask,:]
        y_batch = y[mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      W1, b1 = self.params['W1'], self.params['b1']
      W2, b2 = self.params['W2'], self.params['b2']
      dW1, db1 = grads['W1'], grads['b1']
      dW2, db2 = grads['W2'], grads['b2']
      W1 -= learning_rate*dW1
      W2 -= learning_rate*dW2
      b1 -= learning_rate*np.reshape(db1, db1.shape[1])
      b2 -= learning_rate*np.reshape(db2, db2.shape[1])
    
      self.params['W1'] = W1
      self.params['W2'] = W2
      self.params['b1'] = b1
      self.params['b2'] = b2
    
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X, y=None, reg=0.0)
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


