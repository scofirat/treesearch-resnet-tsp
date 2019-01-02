import tensorflow as tf


class Estimator():
    def __init__(self,dir):
        self.dir=dir
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.__build_graph()
        saver=tf.train.Saver()
        try:
            saver.restore(sess=self.sess,save_path=self.dir+"/model.ckpt")
            print("SaveFile Loaded")
        except ValueError:
            print("Error occurred, initializing variables")
            self.sess.run(tf.global_variables_initializer())
            self.save()

    def residual_block(self, inputLayer, filters, kernel_size):

            shortcut = inputLayer
            residual_layer = tf.layers.conv2d(inputLayer, filters, kernel_size=(kernel_size, kernel_size),
                                              strides=(1, 1), padding='same', use_bias=False)
            residual_layer = tf.layers.batch_normalization(residual_layer, axis=3,
                                                           training=self.is_training)
            residual_layer = tf.nn.relu(residual_layer)
            residual_layer = tf.layers.conv2d(residual_layer, filters, kernel_size=(kernel_size, kernel_size), padding='same',
                                              strides=(1, 1), use_bias=False)
            residual_layer = tf.layers.batch_normalization(residual_layer, axis=3,
                                                           training=self.is_training)
            add_shortcut = tf.add(residual_layer, shortcut)
            residual_result = tf.nn.relu(add_shortcut)
            return residual_result
    """
        Gets the game state, returns the actions by their expected results
    """
    def predict(self,game_state,action_batch):
        return self.sess.run(self.bestMove,{self.game_state:game_state,self.action_batch:action_batch,self.is_training:False})

    def train(self,game_state,action_batch,result):
        return self.sess.run(self.optimize,{self.game_state:game_state,self.action_batch:action_batch,self.result:result,self.is_training:True})

    def best_moves(self,game_state,action_batch):
        return self.sess.run(self.best_moves,{self.game_state:game_state,self.action_batch:action_batch,self.is_training:False})

    def __build_graph(self):
        self.is_training=tf.placeholder(tf.bool)
        self.game_state=tf.placeholder(tf.float32,shape=[1,10,10,8])
        self.action_batch=tf.placeholder(tf.float32,shape=[None,377])
        self.result=tf.placeholder(tf.float32,shape=[None,15])
        layer1 = self.residual_block(self.game_state, filters=8,kernel_size=3)
        layer1 = tf.reshape(layer1, [1, 800])
        layer2 = tf.layers.dense(layer1, 256, kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
        action_vector1=tf.layers.dense(self.action_batch,54, kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
        layer4=tf.tile(layer2,tf.stack([tf.shape(action_vector1)[0],1],axis=0))
        logits=tf.layers.dense(tf.concat([layer4,action_vector1],axis=1),15,activation=tf.math.sigmoid)
        arg_max=tf.argmax(logits,1)
        choice=tf.argmin(arg_max)
        self.bestMove=tf.stack([tf.to_float(choice),tf.to_float(arg_max[choice]),tf.nn.softmax(logits)[choice,arg_max[choice]]],
                                axis=0)
        _,self.best_moves=tf.nn.top_k(-arg_max,k=10)
        loss = tf.losses.sigmoid_cross_entropy(self.result, logits)
        self.optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    def save(self):
        saver=tf.train.Saver()
        saver.save(self.sess,self.dir+"/model.ckpt")