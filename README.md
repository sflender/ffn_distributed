ffn_distributed - a distributed version of Google's ffn (https://github.com/google/ffn), using distributed synchronous stochastic gradient descent, implemented in horovod (https://github.com/uber/horovod).
  
Here are the changes I made with respect to Google's original code:
  
* change model.py to wrap the optimizer inside the horovod distributed optimizer
* all other changes are done inside train.py
* added horovod broadcast operation to synchronize initial model weights
* modified the summary writing process such that rank 0 will produce rank-averaged summaries every 30 steps (customizable)
* only rank 0 will write model checkpoints
* added functionality to decay the learning rate, controlled by decay_learning_rate_fraction parameter
* added functionality to scale the learning rate with the number of ranks according to linear or sqrt rule (controlled by scaling_rule parameter)
* added functionality for warm-up of learning rate.