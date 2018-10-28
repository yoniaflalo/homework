import tensorflow as tf
import numpy as np
import tf_util
import load_policy

global args
global observations
global actions
observations = np.load('observations.npy')
actions = np.load('actions.npy').squeeze()

learning_rate = 0.1
batch_size = 128
num_steps = int(observations.shape[0] / batch_size)
num_epoch = 4
display_step = 100

# Network Parameters
n_hidden_1 = 54  # 1st layer number of neurons
n_hidden_2 = 50  # 2nd layer number of neurons
n_hidden_3 = 25
num_input = observations.shape[1]
num_out = actions.shape[1]
tf.reset_default_graph()
# tf Graph input

class my_net:
    def __init__(self):
        self.X = tf.placeholder("float", [None, num_input])
        self.Y = tf.placeholder("float", [None, num_out])

        self.weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, num_out]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'out': tf.Variable(tf.random_normal([num_out]))
        }
        self.logits = self.neural_net(self.X)

    def neural_net(self,x):
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
        out_layer = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return out_layer

    def train(self, sess):
        num_steps = int(observations.shape[0] / batch_size)

        loss_op = tf.reduce_mean(tf.nn.l2_loss(self.logits - self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        accuracy = loss_op

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        saver_train = tf.train.Saver(tf.trainable_variables())


        # Run the initializer
        sess.run(init)
        if tf.train.checkpoint_exists('./tmp/model.ckpt.meta'):
            saver_train.restore(sess, "./tmp/model.ckpt")
        for epoch in range(num_epoch):
            perm = np.random.permutation(observations.shape[0])
            observations_ = observations[perm, :]
            actions_ = actions[perm, :]
            for step in range(0, num_steps + 1):
                batch_x = observations_[step * batch_size:(step + 1) * batch_size, :]
                batch_y = actions_[step * batch_size:(step + 1) * batch_size, :]
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={self.X: batch_x,
                                                                         self.Y: batch_y})
                    print("Epoch {}: Step ".format(epoch) + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")
        saver_train.save(sess, "./tmp/model.ckpt")
        # Calculate accuracy for MNIST test images
        # print("Testing Accuracy:", \
        #       sess.run(accuracy, feed_dict={X: mnist.test.images,
        #                                     Y: mnist.test.labels}))
    def new_policy(self, obs, sess):
        return sess.run(self.logits, feed_dict={self.X: obs})


def run_dagger(sess, net):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    tf_util.initialize()
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    returns = []
    observations2 = []
    actions2 = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = net.new_policy(obs[None,:], sess)
            action_best = policy_fn(obs[None, :])
            observations2.append(obs)
            actions2.append(action_best)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return observations2, actions2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    global args
    args = parser.parse_args()
    net=my_net()
    with tf.Session() as sess:
        net.train(sess)
        global observations
        global actions
        # saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')


        saver = tf.train.Saver(tf.trainable_variables())

        for iter in range(100):
            saver.restore(sess, "./tmp/model.ckpt")
            observations2, actions2 = run_dagger(sess, net)
            observations = np.concatenate((observations, np.array(observations2)))
            actions = np.concatenate((actions, np.array(actions2).squeeze()))
            np.save('observations', observations)
            np.save('actions', actions)
            net.train(sess)




if __name__ == '__main__':
    main()
