import sys
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import pdb

class Wodehouse():
    def __init__(self, training=True):

        with tf.Session() as self.sess:

            # settings
            self.checkpoint_dir = "checkpoints"
            self.model_dir = "export"
            self.ckpt_name = "wodehouse.ckpt"
            self.export_limit = 10000 # iterations after which model is exported

            # load source material
            self.script = './xaa'
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
            self.batch_size = 100

            # Number of characters in a sequence
            self.sequence_length = 50

            # Number of cells in our LSTM layer
            n_cells = 64

            # Number of LSTM layers
            n_layers = 2

            # Total number of characters in the one-hot encoding
            self.n_chars = len(self.vocab)

            self.create_model(n_layers, n_cells, training)


    def create_model(self, n_layers, n_cells, training):

        batch_size = self.batch_size if training else 1
        sequence_length = self.sequence_length if training else 1

        # placeholders
        self.X = tf.placeholder(tf.int32, shape=[None,sequence_length], name='X')
        self.Y = tf.placeholder(tf.int32, shape=[None,sequence_length], name='Y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # we first create a variable to take us from our one-hot representation to our LSTM cells
        embedding = tf.get_variable("embedding", [self.n_chars, n_cells])

        # And then use tensorflow's embedding lookup to look up the ids in X
        Xs = tf.nn.embedding_lookup(embedding, self.X)

        # Split input data into sequence_length list composed of
        # [batch_size, n_cells] arrays
        with tf.name_scope('reslice'):
            Xs = [tf.squeeze(seq, [1])
                    for seq in tf.split(1, sequence_length, Xs)]

        # Connect timesteps to LSTM cells
        cells = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_cells, state_is_tuple=True, forget_bias=0.6)

        # Set initial_size at all zeros
        self.initial_state = cells.zero_state(tf.shape(self.X)[0], tf.float32)

        # Create layers 2->n_layers
        cells = tf.nn.rnn_cell.MultiRNNCell(
            [cells] * n_layers, state_is_tuple=True)

        # Reinitialize initial_state using new depth of cells
        self.initial_state = cells.zero_state(tf.shape(self.X)[0], tf.float32)

        cells = tf.nn.rnn_cell.DropoutWrapper(cells, output_keep_prob=self.keep_prob)

        # Create RNN using cells, input data and initial_state
        outputs, self.final_state = tf.nn.rnn(cell=cells, inputs=Xs, initial_state=self.initial_state)

        # Flatten outputs
        outputs_flat = tf.reshape(tf.concat(1, outputs), [-1, n_cells])

        # Create softmax layer for predicting output
        with tf.variable_scope('prediction'):
            W = tf.get_variable(
                        "W",
                        shape=[n_cells, self.n_chars],
                        initializer=tf.random_normal_initializer(stddev=0.1))
            self.b = tf.get_variable(
                        "b",
                        shape=[self.n_chars],
                        initializer=tf.random_normal_initializer(stddev=0.1))

            # Find the output prediction of every single character in our minibatch
            # we denote the pre-activation prediction, logits.
            logits = tf.matmul(outputs_flat, W) + self.b

            # We get the probabilistic version by calculating the softmax of this
            self.probs = tf.nn.softmax(logits)

            # And then we can find the index of maximum probability
            Y_pred = tf.argmax(self.probs, 1)

        # Calculate loss for training
        with tf.variable_scope('loss'):
            # Compute mean cross entropy loss for each output.
            Y_true_flat = tf.reshape(tf.concat(1, self.Y), [-1])
            # logits are [batch_size x timesteps, n_chars] and
            # Y_true_flat are [batch_size x timesteps]
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_true_flat)
            # Compute the mean over our `batch_size` x `timesteps` number of observations
            self.mean_loss = tf.reduce_mean(loss)

        # Create optimizer
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
            gradients = []

            # notice clipping of gradient
            clip = tf.constant(10.0, name="clip")
            for grad, var in optimizer.compute_gradients(self.mean_loss):
                gradients.append((tf.clip_by_value(grad, -clip, clip), var))
            self.updates = optimizer.apply_gradients(gradients)


    # Trainer
    def train(self):

        checkpoint_path = os.path.join(self.checkpoint_dir, self.ckpt_name)

        keep_prob = 0.9
        saver = tf.train.Saver()
        sm = tf.train.SessionManager()
        init = tf.initialize_all_variables()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=self.checkpoint_dir) as sess:

            cursor = 0
            it_i = 0
            while it_i < self.export_limit:
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

                loss_val, _ = sess.run([self.mean_loss, self.updates], feed_dict={self.X: Xs, self.Y: Ys, self.keep_prob: keep_prob})
                if it_i % 10 == 0:
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
                        print('synth(amax):', "".join(
                            [self.decoder[ch] for ch in p]))
                    else:
                        print([decoder[ch] for ch in ps])

                it_i += 1
            if not os.path.isdir(self.model_dir):
                os.makedirs(model_dir)
                saver.save(sess, os.path.join(model_dir, 'export'))

    def restore(self):
        new_saver = tf.train.import_meta_graph('export/export.meta')
        new_saver.restore(self.sess, 'export/export')
        print("Model restored.")


    def infer(self, start_char=' '):
        #pdb.set_trace()
        temperature = 0.5
        curr_states = None
        n_iterations = 100 # length of string we're trying to infer

        init = tf.initialize_all_variables()
        self.sess.run(init)

        # Get every tf.Tensor for the initial state
        init_states = []
        for s_i in self.initial_state:
            init_states.append(s_i.c)
            init_states.append(s_i.h)

        # Similarly, for every state after inference
        final_states = []
        for s_i in self.final_state:
            final_states.append(s_i.c)
            final_states.append(s_i.h)

        # Let's start with the letter 't' and see what comes out:
        synth = [[self.encoder[start_char]]]
        for i in range(n_iterations):

            # We'll create a feed_dict parameter which includes what to
            # input to the network, model['X'], as well as setting
            # dropout to 1.0, meaning no dropout.
            feed_dict = {self.X: [synth[-1]]}

            # Now we'll check if we currently have a state as a result
            # of a previous inference, and if so, add to our feed_dict
            # parameter the mapping of the init_state to the previous
            # output state stored in "curr_states".
            if curr_states:
                feed_dict.update(
                    {init_state_i: curr_state_i
                     for (init_state_i, curr_state_i) in
                         zip(init_states, curr_states)})

            # Now we can infer and see what letter we get
            p = self.sess.run(self.probs, feed_dict=feed_dict)[0]

            # And make sure we also keep track of the new state
            curr_states = self.sess.run(final_states, feed_dict=feed_dict)

            # Find the most likely character
            p = p.astype(np.float64)
            p = np.random.multinomial(1, p.ravel() / p.sum())
            p = np.argmax(p)

            # Append to string
            synth.append([p])

            # Print out the decoded letter
            print(self.decoder[p], end='')
            sys.stdout.flush()
