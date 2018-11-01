import os
import gym
import threading
import numpy as np
import multiprocessing
import tensorflow as tf

global global_rewards
global global_episodes
LOG_DIR = 'log_dir'
MODEL_PATH = "saved_model/"
UPDATE_GLOBAL_ITER = 10
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
N_WORKERS = multiprocessing.cpu_count()

class GameEnv:
    def __init__(self):
        self.env = gym.make('Pendulum-v0').unwrapped
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.reward_space = 1
        self.action_bond = [self.env.action_space.low, self.env.action_space.high]

class ActorCritic:
    def __init__(self, name, scope, sess, env, master=None):
        self.sess = sess
        self.name = name
        self.game_env = env
        self.master = master
        self.entropy_beta = 0.01
        scope = scope
        self.actor_optimizer = tf.train.RMSPropOptimizer(0.0001, name='RMSPropA')
        self.critic_optimizer = tf.train.RMSPropOptimizer(0.001, name='RMSPropC')

        with tf.variable_scope(scope):
            self.state_ph = tf.placeholder(tf.float32, [None, env.state_space], self.name+'input_st_ph')
            self.action_ph = tf.placeholder(tf.float32, [None, env.action_space], self.name + 'input_at_ph')
            self.value_target = tf.placeholder(tf.float32, [None, env.reward_space], self.name + 'input_v_ph')

            self.mu, self.sigma, self.value, self.a_params, self.c_params = self.net(self.state_ph, scope)

            td = tf.subtract(self.value_target, self.value, name='TD_error')
            self.c_loss = self.get_critic_loss(td)
            self.pi = self.get_policy(self.mu, self.sigma)
            self.pi_loss = self.get_pi_loss(self.pi, self.action_ph, td)

            with tf.name_scope('choose_a'):
                self.A = tf.clip_by_value(tf.squeeze(self.pi.sample(1), axis=0),
                                              self.game_env.action_bond[0],
                                              self.game_env.action_bond[1])
            with tf.name_scope('local_grad'):
                self.a_grads = tf.gradients(self.pi_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            if "global" in self.name:
                global_actor_vars = self.get_actor_params()
                global_critic_var = self.get_critic_params()
            else:
                global_actor_vars = self.master.get_actor_params()
                global_critic_var = self.master.get_critic_params()

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.a_params, global_actor_vars)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.c_params, global_critic_var)]

                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads,
                                                                                global_actor_vars))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads,
                                                                                 global_critic_var))

    def get_actor_params(self):
        return self.a_params

    def get_critic_params(self):
        return self.c_params

    def get_action(self, state):
        state = np.reshape(state, [-1, self.game_env.state_space])
        return self.sess.run(self.A, feed_dict={self.state_ph: state})

    def get_value(self, state):
        state = np.reshape(state, [-1, self.game_env.state_space])
        return self.sess.run(self.value, feed_dict={self.state_ph: state})

    def push_weights(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)

    def pull_weights(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def get_pi_loss(self, pi, action, td):
        with tf.name_scope('a_loss'):
            # evaluate pdf of normal distribution at value log(x)
            log_prob = pi.log_prob(action)
            exp_v = log_prob * td
            entropy = pi.entropy()
            exp_v = self.entropy_beta * entropy + exp_v
            a_loss = tf.reduce_mean(-exp_v)
        return a_loss

    def get_policy(self, mu, sigma):
        with tf.name_scope('wrap_a_out'):
            mu, sigma = mu * self.game_env.action_bond[1], sigma + 1e-4
        pi = tf.contrib.distributions.Normal(mu, sigma)
        return pi

    def get_critic_loss(self, td):
        with tf.name_scope('c_loss'):
            c_loss = tf.reduce_mean(tf.square(td))
        return c_loss

    def net(self, state, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            pfc1 = tf.layers.dense(state, 200, tf.nn.relu6, kernel_initializer=w_init, name='pfc1')
            mu = tf.layers.dense(pfc1, self.game_env.action_space, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(pfc1, self.game_env.action_space, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            afc1 = tf.layers.dense(state, 100, tf.nn.relu6, kernel_initializer=w_init, name='afc1')
            v = tf.layers.dense(afc1, 1, kernel_initializer=w_init, name='value')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return mu, sigma, v, a_params, c_params


    def get_params(self):
        params = self.sess.run([self.a_params, self.c_params])
        actor_params = params[0]
        critic_params = params[1]
        print("---")
        for i in range(0, len(actor_params)):
            print(self.a_params[i].name, actor_params[i])
        print(" ")
        for i in range(0, len(critic_params)):
            print(self.c_params[i].name, critic_params[i])
        print("---")

# worker class that inits own environment, trains on it and updloads weights to global net
# for GPU you cant do this
class Worker:
    def __init__(self, name, master, coord, sess):
        self.env = GameEnv()
        self.name = name
        self.coord = coord
        self.master = master
        self.sess = sess
        self.local_ac = ActorCritic(name=name+"local_",
                                    scope=name+"local",
                                    env=self.env,
                                    sess=sess,
                                    master=self.master)

    def work(self, saver):
        global global_rewards
        global global_episodes
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.coord.should_stop() and global_episodes < MAX_GLOBAL_EP:
            st = self.env.env.reset()
            st = np.reshape(st, [-1, self.env.state_space])
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                if 'W_0' in self.name:
                    self.env.env.render()
                at = self.local_ac.get_action(st)
                st1, r, done, info = self.env.env.step(at)
                st1 = np.reshape(st1, [-1, self.env.state_space])
                at = np.reshape(at, [-1, self.env.action_space])

                done = True if ep_t == MAX_EP_STEP - 1 else False
                ep_r += r

                buffer_s.append(st)
                buffer_a.append(at)
                buffer_r.append((r + 8) / 8)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else self.local_ac.get_value(st)

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = np.add(r, np.multiply(0.9, v_s_))
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s = np.reshape(buffer_s, [-1, self.env.state_space])
                    buffer_a = np.reshape(buffer_a, [-1, self.env.action_space])
                    buffer_v_target = np.reshape(buffer_v_target, [-1, self.env.reward_space])

                    feed_dict = {self.local_ac.state_ph: buffer_s,
                                 self.local_ac.action_ph: buffer_a,
                                 self.local_ac.value_target: buffer_v_target}

                    self.local_ac.push_weights(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.local_ac.pull_weights()

                st = st1
                total_step += 1
                if done:
                    if len(global_rewards) < 5:  # record running episode reward
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))  # smoothing
                    print(
                        self.name,
                        "Ep:", global_episodes,
                        "| Ep_r: %i" % global_rewards[-1],
                    )
                    global_episodes += 1

                    break
            if global_episodes % 100 == 0:
                save_path = saver.save(self.sess, MODEL_PATH + "pretrained.ckpt", global_step=global_episodes)
                print("saved to %s" % save_path)

class Controller:
    def __init__(self, sess, mode):
        env = GameEnv()
        coord = tf.train.Coordinator()

        with tf.device("/cpu:0"):
            self.global_ac = ActorCritic(name="global_", scope="global", env=env, sess=sess)
            workers = []
            for i in range(N_WORKERS):
                i_name = 'W_%i' % i  # worker name
                workers.append(Worker(i_name, self.global_ac, coord, sess))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        if mode == "train":
            worker_threads = []
            for worker in workers:  # start workers
                job = lambda: worker.work(saver)
                t = threading.Thread(target=job)
                t.start()
                worker_threads.append(t)
            coord.join(worker_threads)  # wait for termination of workers

        done = False
        st = env.env.reset()
        stcnt = 0
        while not done:
            env.env.render()
            st = np.reshape(st, [-1, env.state_space])
            at = self.global_ac.get_action(st)
            st1, rt, done, info = env.env.step(at)
            print("cos{}: {}, sin{}: {}".format(np.arccos(st1[0]*3.14), st1[0], np.arcsin(st1[1]*3.14), st1[1]))
            print("theta dot {} {}".format(np.arccos(st1[0])*np.arcsin(st1[1]), st1[2]))
            st1 = np.reshape(st1, [-1, env.state_space])
            st = st1
            stcnt += 1
            if stcnt > 1000:
                done = True

def main():
    global global_rewards
    global_rewards = []
    global global_episodes
    global_episodes = 0
    sess = tf.Session()
    Controller(sess, "test")

if __name__ == "__main__":
    main()
