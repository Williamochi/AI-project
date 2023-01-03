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
```
class DRQN():
    def __init__(self, input_shape, num_actions, inital_learning_rate):
        # 初始化所有超參數

        self.tfcast_type = tf.float32
        
        # 設定輸入外形為(length, width, channels)
        self.input_shape = input_shape  
        
        # 環境中的動作數量
        self.num_actions = num_actions
        
        # 神經網路的學習率
        self.learning_rate = inital_learning_rate
                
        # 定義卷積神經網路的超參數

        # 過濾器大小
        self.filter_size = 5
        
        # 過濾器數量
        self.num_filters = [16, 32, 64]
        
        # 間隔大小
        self.stride = 2
        
        # 池大小
        self.poolsize = 2        
        
        # 設定卷積層形狀
        self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]
        
        # 定義循環神經網路與最終前饋層的超參數
        
        # 神經元數量
        self.cell_size = 100
        
        # 隱藏層數量
        self.hidden_layer = 50
        
        # drop out 機率
        self.dropout_probability = [0.3, 0.2]

        # 最佳化的超參數
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180

        
        # 初始化CNN的所有變數

        # 初始化輸入的佔位，形狀為(length, width, channel)
        self.input = tf.compat.v1.placeholder(shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = self.tfcast_type)
        
        # 初始化目標向量的形狀，正好等於動作向量
        self.target_vector = tf.compat.v1.placeholder(shape = (self.num_actions, 1), dtype = self.tfcast_type)

        # 初始化三個回應過濾器的特徵圖
        self.features1 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]),
                                     dtype = self.tfcast_type)
        
        self.features2 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]),
                                     dtype = self.tfcast_type)
                                     
        
        self.features3 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]),
                                     dtype = self.tfcast_type)

        # 初始化RNN變數
        # 討論RNN的運作方式
        
        self.h = tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = self.tfcast_type)
        
        # 隱藏層對隱藏層的權重矩陣
        self.rW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            high = np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            size = (self.convolution_shape, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # 輸入層對隱藏層的權重矩陣
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
        # 偏差
        self.rb = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)
        self.rc = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)

        
        # 定義前饋網路的權重與偏差
        
        # 權重
        self.fW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            high = np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            size = (self.cell_size, self.num_actions)),
                              dtype = self.tfcast_type)
                             
        # 偏差
        self.fb = tf.Variable(initial_value = np.zeros(self.num_actions), dtype = self.tfcast_type)

        # 學習率
        self.step_count = tf.Variable(initial_value = 0, dtype = self.tfcast_type)
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate,       
                                                   self.step_count,
                                                   self.loss_decay_steps,
                                                   self.loss_decay_steps,
                                                   staircase = False)
        
        
        # 建置網路

        # 第一卷積層
        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), filter = self.features1, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = tf.nn.max_pool2d(self.relu1, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # 第二卷積層
        self.conv2 = tf.nn.conv2d(input = self.pool1, filter = self.features2, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = tf.nn.max_pool2d(self.relu2, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # 第三卷積層
        self.conv3 = tf.nn.conv2d(input = self.pool2, filter = self.features3, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu3 = tf.nn.relu(self.conv3)
        self.pool3 = tf.nn.max_pool2d(self.relu3, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # 加入 dropout 並重新設定輸入外形
        self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0])
        self.reshaped_input = tf.reshape(self.drop1, shape = [1, -1])


        # 建置循環神經網路，會以卷積網路的最後一層作為輸入
        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb)
        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)

        # 在RNN中加入dropout
        self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1])
        
        # 將RNN的結果送給前饋層
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape = [-1, 1])
        self.prediction = tf.argmax(self.output)

        # 計算損失
        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))
        
        # 使用 Adam 最佳器將誤差降到最低
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        
        # 計算損失的梯度並更新梯度
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)

        self.parameters = (self.features1, self.features2, self.features3,
                           self.rW, self.rU, self.rV, self.rb, self.rc,
                           self.fW, self.fb)
```
```
class ExperienceReplay():
    def __init__(self, buffer_size):
        
        # 儲存轉移的緩衝
        self.buffer = []       
        
        # 緩衝大小
        self.buffer_size = buffer_size
        
    # 如果緩衝滿了就移除舊的轉移
    # 可把緩衝視佇列，新的進來時，舊的就出去
    
    def appendToBuffer(self, memory_tuplet):
        if len(self.buffer) > self.buffer_size: 
            for i in range(len(self.buffer) - self.buffer_size):
                self.buffer.remove(self.buffer[0])     
        self.buffer.append(memory_tuplet)  
        
        
    # 定義 sample 函式來隨機取樣n個轉移  
    
    def sample(self, n):
        memories = []
        for i in range(n):
            memory_index = np.random.randint(0, len(self.buffer))       
            memories.append(self.buffer[memory_index])
        return memories
```
```
def train(num_episodes, episode_length, learning_rate, scenario = "deathmatch.cfg", map_path = 'map02', render = False):
  
    # discount parameter for Q-value computation
    discount_factor = .99
    
    # frequency for updating the experience in the buffer
    update_frequency = 5
    store_frequency = 50
    
    # for printing the output
    print_frequency = 1000

    # initialize variables for storing total rewards and total loss
    total_reward = 0
    total_loss = 0
    old_q_value = 0

    # initialize lists for storing the episodic rewards and losses 
    rewards = []
    losses = []

    # okay, now let us get to the action!
   
    # first, we initialize our doomgame environment
    game = viz.DoomGame()
    
    # specify the path where our scenario file is located
    game.set_doom_scenario_path(scenario)
    
    # specify the path of map file
    game.set_doom_map(map_path)

    # then we set screen resolution and screen format
    game.set_screen_resolution(viz.ScreenResolution.RES_256X160)    
    game.set_screen_format(viz.ScreenFormat.RGB24)

    # we can add particles and effetcs we needed by simply setting them to true or false
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)

    # now we will specify buttons that should be available to the agent
    game.add_available_button(viz.Button.MOVE_LEFT)
    game.add_available_button(viz.Button.MOVE_RIGHT)
    game.add_available_button(viz.Button.TURN_LEFT)
    game.add_available_button(viz.Button.TURN_RIGHT)
    game.add_available_button(viz.Button.MOVE_FORWARD)
    game.add_available_button(viz.Button.MOVE_BACKWARD)
    game.add_available_button(viz.Button.ATTACK)
    
   
    # okay,now we will add one more button called delta. The above button will only work 
    # like a keyboard keys and will have only boolean values. 

    # so we use delta button which emulates a mouse device which will have positive and negative values
    # and it will be useful in environment for exploring
    
    game.add_available_button(viz.Button.TURN_LEFT_RIGHT_DELTA, 90)
    game.add_available_button(viz.Button.LOOK_UP_DOWN_DELTA, 90)

    # initialize an array for actions
    actions = np.zeros((game.get_available_buttons_size(), game.get_available_buttons_size()))
    count = 0
    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()


    # then we add the game variables, ammo, health, and killcount
    game.add_available_game_variable(viz.GameVariable.AMMO0)
    game.add_available_game_variable(viz.GameVariable.HEALTH)
    game.add_available_game_variable(viz.GameVariable.KILLCOUNT)

    # we set episode_timeout to terminate the episode after some time step
    # we also set episode_start_time which is useful for skipping intial events
    
    game.set_episode_timeout(6 * episode_length)
    game.set_episode_start_time(10)
    game.set_window_visible(render)
    
    # we can also enable sound by setting set_sound_enable to true
    game.set_sound_enabled(False)

    # we set living reward to 0 which the agent for each move it does even though the move is not useful
    game.set_living_reward(0)

    # doom has different modes such as player, spectator, asynchronous player and asynchronous spectator
    
    # in spectator mode humans will play and agent will learn from it.
    # in player mode, agent actually plays the game, so we use player mode.
    
    game.set_mode(viz.Mode.PLAYER)

    # okay, So now we, initialize the game environment
    game.init()

    # now, let us create instance to our DRQN class and create our both actor and target DRQN networks
    actionDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
    targetDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
    
    # we will also create instance to the ExperienceReplay class with the buffer size of 1000
    experiences = ExperienceReplay(1000)

    # for storing the models
    saver = tf.compat.v1.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    
    # now let us start the training process
    # we initialize variables for sampling and storing transistions from the experience buffer
    sample = 5
    store = 50
   
    # start the tensorflow session
    with tf.compat.v1.Session() as sess:
        
        # initialize all tensorflow variables
        
        sess.run(tf.global_variables_initializer())
        
        for episode in range(num_episodes):
            
            # start the new episode
            game.new_episode()
            
            # play the episode till it reaches the episode length
            for frame in range(episode_length):
                
                # get the game state
                state = game.get_state()
                s = state.screen_buffer
                
                # select the action
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
                action = actions[a]
                
                # perform the action and store the reward
                reward = game.make_action(action)
                
                # update total rewad
                total_reward += reward

               
                # if the episode is over then break
                if game.is_episode_finished():
                    break
                 
                # store transistion to our experience buffer
                if (frame % store) == 0:
                    experiences.appendToBuffer((s, action, reward))

                # sample experience form the experience buffer        
                if (frame % sample) == 0:
                    memory = experiences.sample(1)
                    mem_frame = memory[0][0]
                    mem_reward = memory[0][2]
                    
                    
                    # now, train the network
                    Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame})
                    Q2 = targetDRQN.output.eval(feed_dict = {targetDRQN.input: mem_frame})

                    # set learning rate
                    learning_rate = actionDRQN.learning_rate.eval()

                    # calculate Q value
                    Qtarget = old_q_value + learning_rate * (mem_reward + discount_factor * Q2 - old_q_value)    
                    
                    # update old Q value
                    old_q_value = Qtarget

                    # compute Loss
                    loss = actionDRQN.loss.eval(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    
                    # update total loss
                    total_loss += loss

                    # update both networks
                    actionDRQN.update.run(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    targetDRQN.update.run(feed_dict = {targetDRQN.target_vector: Qtarget, targetDRQN.input: mem_frame})

            rewards.append((episode, total_reward))
            losses.append((episode, total_loss))

            
            print("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))


            total_reward = 0
            total_loss = 0
```
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

