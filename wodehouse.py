import sys
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

class Wodehouse():
    def __init__(self):
        # load source material
        self.script = './wodehouse.txt'
        self.txts = []
        with open(self.script, 'r', encoding="utf-8") as fp:
            self.txt = fp.read()
        self.txt = "\n".join([txt_i.strip()
                         for txt_i in self.txt.replace('\t', '').split('\n')
                         if len(txt_i)])

        # vocabulary
        self.vocab = list(set(self.txt))
        self.vocab.sort()

        # encoder and decoder
        self.encoder = OrderedDict(zip(self.vocab, range(len(self.vocab))))
        self.decoder = OrderedDict(zip(range(len(self.vocab)), self.vocab))

        # Number of sequences in a mini batch
        self.batch_size = 50

        # Number of characters in a sequence
        self.sequence_length = 50

        # Number of cells in our LSTM layer
        self.n_cells = 128

        # Number of LSTM layers
        self.n_layers = 2

        # Total number of characters in the one-hot encoding
        self.n_chars = len(self.vocab)

        # placeholders
        self.X = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name='X')
        self.Y = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name='Y')

        # we first create a variable to take us from our one-hot representation to our LSTM cells
        self.embedding = tf.get_variable("embedding", [self.n_chars, self.n_cells])

        # And then use tensorflow's embedding lookup to look up the ids in X
        self.Xs = tf.nn.embedding_lookup(self.embedding, self.X)

        # Split input data into sequence_length list composed of
        # [batch_size, n_cells] arrays
        with tf.name_scope('reslice'):
            self.Xs = [tf.squeeze(seq, [1])
                  for seq in tf.split(1, self.sequence_length, self.Xs)]

        # Connect timesteps to LSTM cells
        self.cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_cells, state_is_tuple=True, forget_bias=1.0)

        # Set initial_size at all zeros
        self.initial_state = self.cells.zero_state(tf.shape(self.X)[0], tf.float32)

        # Create layers 2->n_layers
        self.cells = tf.nn.rnn_cell.MultiRNNCell(
            [self.cells] * self.n_layers, state_is_tuple=True)

        # Reinitialize initial_state using new depth of cells
        self.initial_state = self.cells.zero_state(tf.shape(self.X)[0], tf.float32)

        # Create RNN using cells, input data and initial_state
        self.outputs, self.state = tf.nn.rnn(cell=self.cells, inputs=self.Xs, initial_state=self.initial_state)

        # Flatten outputs
        self.outputs_flat = tf.reshape(tf.concat(1, self.outputs), [-1, self.n_cells])

        # Create softmax layer for predicting output
        with tf.variable_scope('prediction'):
            self.W = tf.get_variable(
                        "W",
                        shape=[self.n_cells, self.n_chars],
                        initializer=tf.random_normal_initializer(stddev=0.1))
            self.b = tf.get_variable(
                        "b",
                        shape=[self.n_chars],
                        initializer=tf.random_normal_initializer(stddev=0.1))

            # Find the output prediction of every single character in our minibatch
            # we denote the pre-activation prediction, logits.
            self.logits = tf.matmul(self.outputs_flat, self.W) + self.b

            # We get the probabilistic version by calculating the softmax of this
            self.probs = tf.nn.softmax(self.logits)

            # And then we can find the index of maximum probability
            self.Y_pred = tf.argmax(self.probs, 1)

        # Calculate loss for training
        with tf.variable_scope('loss'):
            # Compute mean cross entropy loss for each output.
            self.Y_true_flat = tf.reshape(tf.concat(1, self.Y), [-1])
            # logits are [batch_size x timesteps, n_chars] and
            # Y_true_flat are [batch_size x timesteps]
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.Y_true_flat)
            # Compute the mean over our `batch_size` x `timesteps` number of observations
            self.mean_loss = tf.reduce_mean(self.loss)

        # Create optimizer
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.gradients = []

            # notice clipping of gradient
            self.clip = tf.constant(5.0, name="clip")
            for grad, var in self.optimizer.compute_gradients(self.mean_loss):
                self.gradients.append((tf.clip_by_value(grad, -self.clip, self.clip), var))
            self.updates = self.optimizer.apply_gradients(self.gradients)


    # Trainer
    def train(self):

        checkpoint_dir = "checkpoints"
        model_dir = "export"
        ckpt_name = "wodehouse.ckpt"
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

        saver = tf.train.Saver()
        sm = tf.train.SessionManager()
        init = tf.initialize_all_variables()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=checkpoint_dir) as sess:

            cursor = 0
            it_i = 0
            while it_i < 1000001:
                Xs, Ys = [], []
                for batch_i in range(self.batch_size):
                    if (cursor + self.sequence_length) >= len(self.txt) - self.sequence_length - 1:
                        cursor = 0
                    Xs.append([self.encoder[ch]
                               for ch in self.txt[cursor:cursor + self.sequence_length]])
                    Ys.append([self.encoder[ch]
                               for ch in self.txt[cursor + 1: cursor + self.sequence_length + 1]])

                    cursor = (cursor + self.sequence_length)
                Xs = np.array(Xs).astype(np.int32)
                Ys = np.array(Ys).astype(np.int32)

                loss_val, _ = sess.run([self.mean_loss, self.updates], feed_dict={self.X: Xs, self.Y: Ys})
                if it_i % 100 == 0:
                    print(it_i, loss_val)

                if it_i % 1000 == 0:
                    # Save the variables to disk.
                    save_path = saver.save(sess, checkpoint_path, global_step=it_i)
                    print("Model saved in file: %s" % save_path)

                    p = sess.run(self.probs, feed_dict={self.X: Xs})
                    ps = [np.random.choice(range(self.n_chars), p=p_i.ravel())
                          for p_i in p]
                    p = [np.argmax(p_i) for p_i in p]
                    if isinstance(self.txt[0], str):
                        print('original:', "".join(
                            [self.decoder[ch] for ch in Xs[-1]]))
                        print('synth(samp):', "".join(
                            [self.decoder[ch] for ch in ps]))
                        print('synth(amax):', "".join(
                            [self.decoder[ch] for ch in p]))
                    else:
                        print([decoder[ch] for ch in ps])

                it_i += 1
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
                saver.save(sess, os.path.join(model_dir, 'export'))
