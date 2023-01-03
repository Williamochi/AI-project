---
layout: post
title: Capstone-project
author: [mochi_pancake_elvisting]
category: [Lecture]
tags: [jekyll, ai]
---

期末專題實作:運用DRQN及C15訓練遊戲之強化學習

---
## 運用DRQN及C15訓練遊戲之強化學習
### 組員
00953150 鄭丞恩  
00953128 丁昱鈞  
00953101 李承恩
### 系統簡介及功能說明
# **系統簡介**:
初步想法:  
1.做出地圖並訓練狗狗走出ex:單一出口的封閉空間  
2.隨機丟出物品並讓狗狗撿取並回到原位
# **功能說明**:
---
### 系統方塊圖

演算法模型說明
class DRQN():
    def __init__(self, input_shape, num_actions, inital_learning_rate):
        
        # first, we initialize all the hyperparameters

        self.tfcast_type = tf.float32
        
        # shape of our input which would be (length, width, channels)
        self.input_shape = input_shape  
        
        # number of actions in the environment
        self.num_actions = num_actions
        
        # learning rate for the neural network
        self.learning_rate = inital_learning_rate
                
        # now we will define the hyperparameters of the convolutional neural network 

        # filter size
        self.filter_size = 5
        
        # number of filters
        self.num_filters = [16, 32, 64]
        
        # stride size
        self.stride = 2
        
        # pool size
        self.poolsize = 2        
        
        # shape of our convolutional layer
        self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]
        
        # now we define the hyperparameters of our recurrent neural network and the final feed forward layer
        
        # number of neurons 
        self.cell_size = 100
        
        # number of hidden layers
        self.hidden_layer = 50
        
        # drop out probability
        self.dropout_probability = [0.3, 0.2]

        # hyperparameters for optimization
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180

        
        # initialize all the variables for the CNN

        # we initialize the placeholder for input whose shape would be (length, width, channel)
        self.input = tf.compat.v1.placeholder(shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = self.tfcast_type)
        
        # we will also initialize the shape of the target vector whose shape is equal to the number of actions
        self.target_vector = tf.compat.v1.placeholder(shape = (self.num_actions, 1), dtype = self.tfcast_type)

        # initialize feature maps for our corresponding 3 filters
        self.features1 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]),
                                     dtype = self.tfcast_type)
        
        self.features2 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]),
                                     dtype = self.tfcast_type)
                                     
        
        self.features3 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]),
                                     dtype = self.tfcast_type)

        # initialize variables for RNN
        # recall how RNN works from chapter 7
        
        self.h = tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = self.tfcast_type)
        
        # hidden to hidden weight matrix
        self.rW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            high = np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            size = (self.convolution_shape, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # input to hidden weight matrix
        self.rU = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # hiddent to output weight matrix
                          
        self.rV = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)
        # bias
        self.rb = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)
        self.rc = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)

        
        # initialize weights and bias of feed forward network
        
        # weights
        self.fW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            high = np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            size = (self.cell_size, self.num_actions)),
                              dtype = self.tfcast_type)
                             
        # bias
        self.fb = tf.Variable(initial_value = np.zeros(self.num_actions), dtype = self.tfcast_type)

        # learning rate
        self.step_count = tf.Variable(initial_value = 0, dtype = self.tfcast_type)
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate,       
                                                   self.step_count,
                                                   self.loss_decay_steps,
                                                   self.loss_decay_steps,
                                                   staircase = False)
        
        
        # now let us build the network

        # first convolutional layer
        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), filter = self.features1, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = tf.nn.max_pool2d(self.relu1, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # second convolutional layer
        self.conv2 = tf.nn.conv2d(input = self.pool1, filter = self.features2, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = tf.nn.max_pool2d(self.relu2, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # third convolutional layer
        self.conv3 = tf.nn.conv2d(input = self.pool2, filter = self.features3, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu3 = tf.nn.relu(self.conv3)
        self.pool3 = tf.nn.max_pool2d(self.relu3, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # add dropout and reshape the input
        self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0])
        self.reshaped_input = tf.reshape(self.drop1, shape = [1, -1])


        # now we build recurrent neural network which takes the input from the last layer of convolutional network
        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb)
        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)

        # add drop out to RNN
        self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1])
        
        # we feed the result of RNN to the feed forward layer
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape = [-1, 1])
        self.prediction = tf.argmax(self.output)

        # compute loss
        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))
        
        # we use Adam optimizer for minimizing the error
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        
        # compute gradients of the loss and update the gradients
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)

        self.parameters = (self.features1, self.features2, self.features3,
                           self.rW, self.rU, self.rV, self.rb, self.rc,
                           self.fW, self.fb)
---
### 製作步驟
1. 建立資料集dataset
2. 移植程式到kaggle
3. kaggle訓練模型
4. kaggle測試模型
---
### 系統測試及成果展示

---



<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

