import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = np.shape(x)[0]
  M = np.shape(w)[1]
  x_reshaped = np.reshape(x,(N,-1))
  D = np.shape(x_reshaped)[1]
  out = x_reshaped.dot(w) + b
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  N = np.shape(x)[0]
  D = np.shape(w)[0]
  M = np.shape(w)[1]
  #print "N is %d, D is %d, M is %d"%(N,D,M)
    
  x_reshape = np.reshape(x, (N, -1))
    
  dx = np.dot(dout, w.T)
  dx = np.reshape(dx, np.shape(x))
    
  dw = np.dot(x_reshape.T, dout)

  db = np.sum(dout, axis = 0)

  # x1  x2  w1  w2  =  x1w1+x2w3  x1w2+x2w4 = y1  y2
  # x3  x4  w3  w4     x3w1+x4w3  x3w2+x4w4 = y3  y4
  # x5  x6             x5w1+x6w3  x5w2+x6w4 = y5  y6
  # dy1/dx1 = w1 dy2/dx1 = w2
  # dx1 = dy1*w1 + dy2*w2
  # dx  =   dy1*w1+dy2*w2   dy1*w3+dy2*w4
  #         dy3*w1+dy4*w2   dy3*w3+dy4*w4
  #         dy5*w1+dy6*w2   dy5*w3+dy6*w4
  #         dx = dout*wT
  # dw = x1+x3+x5  x1+x3+x5
  #      x2+x4+x6  x2+x4+x6
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #print "dw's shape is (%d,%d)"%(np.shape(dw)[0], np.shape(dw)[1])
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(x,0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  pos_idx = (x>0)
  neg_idx = (x<=0)
  dx_reLu = np.zeros(np.shape(x))
  dx_reLu[pos_idx] = 1
  dx_reLu[neg_idx] = 0
  dx = np.multiply(dx_reLu, dout)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis=0, keepdims=True) #(D,1)
    sample_var  = np.var(x, axis=0, keepdims=True)  #(D,1)
    xu = x - sample_mean # (N,D) - (D,1) -> (N,D)
    sqrtvar = np.sqrt(sample_var + eps) #(D,1)
    ivar = 1/sqrtvar #(D,1)
    x_normed = xu*ivar  # (N,D) - (D,1) -> (N,D)
    out = gamma*x_normed + beta
    running_mean = momentum*running_mean + (1-momentum)*sample_mean
    running_var = momentum*running_var + (1-momentum)* sample_var
    
    cache = sample_mean, sample_var, eps,  x_normed, running_mean, running_var, gamma, beta, xu, sqrtvar, ivar

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x-= running_mean
    x_normed = x/np.sqrt(running_var + eps)
    out = gamma*x_normed + beta
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N, D = np.shape(dout)
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  sample_mean, sample_var, eps,  x_normed, running_mean, running_var, gamma, beta, xu, sqrtvar, ivar = cache

  
  dgamma = np.sum(dout*x_normed, axis=0,keepdims = True)
  dbeta = np.sum(dout, axis=0, keepdims = True)
  #print "The shape of dgamma is (%d,%d)"%(np.shape(dgamma)[0], np.shape(dgamma)[1])
  #print "The shape of dbeta is (%d,%d)"%(np.shape(dbeta)[0], np.shape(dbeta)[1])
  
  dL_dnormed = np.multiply(gamma, dout) #(N,D)
  #xu = x-u
  dL_xu1 = dL_dnormed*ivar #(N,D)/(D,1)
  #ivar = 1/var
  #var = sqrt(sample_Var + eps)
  #sample_var = 1/D(Sigma xu)

  dL_divar = np.sum(np.multiply(dL_dnormed, xu), axis=0,keepdims=True)      #(D,1)
  #print "The shape of dL_divar is (%d,%d)"%(np.shape(dL_divar)[0], np.shape(dL_divar)[1])
  dL_dsqrtvar = -1./(sqrtvar**2)*dL_divar                          #(D,1)
  dL_dvar = dL_dsqrtvar * (0.5/sqrtvar)   #(D,1)
  dsq = 1./N*np.ones((N,D))*dL_dvar #(N,D)
  dL_dxu2 = dsq * 2 * xu         #(N,D)

  dL_dx1 =  dL_xu1 + dL_dxu2  #(N,D)
  dL_du = (-1)*np.sum(dL_xu1 + dL_dxu2, axis = 0)
  dL_dx2 =1./N*np.ones((N,D))*dL_du

  dx = dL_dx1 + dL_dx2

  #print "The shape of dx is (%d,%d)"%(np.shape(dx)[0], np.shape(dx)[1])
  
    
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p)/p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  stride = conv_param['stride']
  pad = conv_param['pad']
  N,C,H,W = np.shape(x)
  F,C,HH,WW = np.shape(w)
  if ((H + 2 * pad - HH) % stride != 0) or ((W + 2 * pad - WW) % stride!=0) :
     print "Convolution can't fit"
     assert(0)
    
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros([N,F,H_out, W_out])
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  for i in xrange(N):
    X_PAD = np.lib.pad(x[i,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    #print np.shape(X_PAD)
    #print np.shape(w[i,:,:,:])
    for j in xrange(F):
       for k in xrange(H_out):
          for l in xrange(W_out):
             out[i,j,k,l]=np.sum(np.multiply(X_PAD[:, k*stride:k*stride+HH, l*stride:l*stride+WW], w[j,:,:,:]))+b[j]
             #print out
            
  #print out
        
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db, dX_PAD = None, None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x,w,b,conv_param = cache
  
  stride = conv_param['stride']
  pad = conv_param['pad']
  N,C,H,W = np.shape(x)
  F,C,HH,WW = np.shape(w)
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  dX_PAD = np.zeros([N,C,H+2*pad,W+2*pad])
  for i in xrange(N):
    X_PAD = np.lib.pad(x[i,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    for j in xrange(F):
      for k in xrange(H_out):
        for l in xrange(W_out):
          dX_PAD[i,:, k*stride:k*stride+HH, l*stride:l*stride+WW]+=w[j,:,:,:]*dout[i,j,k,l]
          dw[j,:,:,:]+=dout[i,j,k,l]*X_PAD[:, k*stride:k*stride+HH, l*stride:l*stride+WW]
          db[j]+=dout[i,j,k,l]
        
  
  dx = dX_PAD[:,:,pad:pad+H, pad:pad+W]
     
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  N,C,H,W = np.shape(x)
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  if (H-pool_height)%stride != 0 and (W-pool_width)%stride != 0 :
    print "Max Pooling size doesn't fit"
    assert(0)
  H_out = (H-pool_height)/stride + 1
  W_out = (W-pool_width)/stride + 1
  out = np.zeros([N,C,H_out, W_out])
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  for i in xrange(N):
    for j in xrange(C):
      for k in xrange(H_out):
        for l in xrange(W_out):
          out[i,j,k,l]=  np.amax(x[i,j, k*stride:k*stride+pool_height, l*stride:l*stride+pool_width ])
          
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  N,C,H,W = np.shape(x)
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  H_out = (H-pool_height)/stride + 1
  W_out = (W-pool_width)/stride + 1
  dx = np.zeros_like(x)
  max_idx = np.zeros_like(dout)
  
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  for i in xrange(N):
    for j in xrange(C):
      for k in xrange(H_out):
        for l in xrange(W_out):
          max_in_region = np.amax(x[i,j, k*stride:k*stride+pool_height, l*stride:l*stride+pool_width ])
          max_mask = (x[i,j, k*stride:k*stride+pool_height, l*stride:l*stride+pool_width] == max_in_region)
          dx[i,j,k*stride:k*stride+pool_height, l*stride:l*stride+pool_width]+=dout[i,j,k,l]*max_mask
            

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  N,C,H,W = np.shape(x)

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  x_shaped = x.transpose(0,2,3,1).reshape([N*H*W,C])
  out_shaped, cache = batchnorm_forward(x_shaped, gamma, beta, bn_param)
  out = out_shaped.reshape([N,H,W,C]).transpose(0,3,1,2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None
  N,C,H,W = np.shape(dout)
  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  dout_shaped = dout.transpose(0,2,3,1).reshape([N*H*W,C])
  dx_shaped, dgamma, dbeta = batchnorm_backward(dout_shaped, cache)
  dx = dx_shaped.reshape([N,H,W,C]).transpose(0,3,1,2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
