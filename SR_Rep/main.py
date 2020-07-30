from defs import *
from models import *
from utils import *
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import gym
import random

class network():
	def __init__(self, sess):
		self.sess = sess
		self.devs = device_lib.list_local_devices()
		data = dict()
		self.net_details = dict()
		with tf.name_scope("Inputs"):
			self.input_st_ph = tf.placeholder("float", [None, 1, 4], name="input_st")
			self.input_stimg_ph = tf.placeholder("float", ST_SHAPES, name="input_st_image")
			self.input_at_ph = tf.placeholder("float", AT_SHAPES, name="input_at")
			self.input_pi_ph = tf.placeholder("float", [None, 1, 4], "input_pi")
			self.input_adv = tf.placeholder("float", ADV_SHAPES, "input_adv")
			self.input_returns = tf.placeholder("float", ADV_SHAPES, "input_returns")
			self.input_st_latents = tf.placeholder("float", ST_LT_SHAPES, name="input_state_latents")

		with tf.name_scope("Phi"):
			stendec = state_encoder_decoder("Phi", LATENT_DIM)
			'''
			with tf.name_scope("Phi_Encode"):
				self.phi = stendec.encode(self.input_st_ph, False)
				data["phi_input"] = self.input_st_ph
			'''
			with tf.name_scope("Phi_Decode"):
				self.st_ = stendec.decode(self.input_st_ph, False)
				data["phi_output"] = self.st_
				self.st_recon = stendec.decode(self.input_st_latents, True)
				
			with tf.name_scope("Phi_loss"):
				self.phi_loss = stendec.get_loss(source=self.st_,
												target=self.input_st_ph)
				data["phi_loss"] = self.phi_loss
				
			with tf.name_scope("Phi_train"):
				self.phi_train_step = stendec.train_step(self.phi_loss)
			self.net_details["phi"] = stendec.get_vars()
		
		with tf.name_scope("Theta"):
			atencdec = action_encode_decoder("Theta")
			with tf.name_scope("Theta_Encode"):
				self.theta = atencdec.encode(self.input_at_ph, False)
			with tf.name_scope("Theta_Decode"):
				self.at_ = atencdec.decode(self.theta, False)
			with tf.name_scope("Theta_loss"):
				self.theta_loss = atencdec.get_loss(source=self.at_,
												    target=self.input_at_ph)
				data["theta_loss"] = self.theta_loss
			with tf.name_scope("Theta_train"):
				self.theta_train_step = atencdec.train_step(self.theta_loss)
			self.net_details["theta"] = atencdec.get_vars()
		
		with tf.name_scope("critic"):
			vs = critic("critic_")
			self.st_value = vs.get_value(self.input_st_ph, False)
			self.critic_loss = vs.get_loss(value=self.st_value,
										   target_value=self.input_returns)
			data["critic_loss"] = self.critic_loss
			self.critic_train_step = vs.train_step(self.critic_loss)
			self.net_details["critic"] = vs.get_vars()
			
		with tf.name_scope("pi"):
			with tf.device('/device:GPU:1'):
				pi = policy("pi_")
				self.pi_out = pi.get_action(self.input_pi_ph, False)
				self.pi_grads, self.pi_scaled_grads = pi.get_scaled_grads(self.pi_out, self.input_adv, 1)
				self.pi_train_step = pi.train_step(self.pi_scaled_grads)
				self.net_details["pi"] = pi.get_vars()
		
		tensorboard_summary(data)
		with tf.name_scope("Tensorboard"):
			self.merged = tf.summary.merge_all()

	'''
	def get_state_latent(self, input_st):
		input_st = np.reshape(input_st, ST_SHAPE)
		latent = self.sess.run(self.phi, feed_dict={self.input_st_ph: input_st})
		return latent
	'''
	
	def get_state_reconstruction(self, state_latent):
		state_latent = np.reshape(state_latent, (-1, LATENT_DIM))
		recon = self.sess.run(self.st_recon, feed_dict={self.input_st_latents: state_latent})
		return recon
	
	def get_action(self, state_latent):
		state_latent = np.reshape(state_latent, (-1, LATENT_DIM))
		action = self.sess.run(self.pi_out, feed_dict={self.input_pi_ph: state_latent})
		return action
	
	def get_state_values(self, state_latent):
		state_latent = np.reshape(state_latent, (-1, LATENT_DIM))
		value = self.sess.run(self.st_value, feed_dict={self.input_st_latents: state_latent})
		return value
	
	def train_step_sr(self, memory):
		s_batch, slat_batch, a_batch, r_batch, s2_batch, slat1_batch, done_batch = memory.sample_batch(10)
		feed_dict = {self.input_st_ph: slat_batch,
					 self.input_stimg_ph: s_batch,
					 self.input_at_ph: a_batch,
					 self.input_returns: r_batch}
		
		ops = []
		ops.append(self.phi_train_step)
		ops.append(self.theta_train_step)
		ops.append(self.critic_train_step)
		self.sess.run(ops, feed_dict=feed_dict)
		
		ops = []
		ops.append(self.phi_loss)
		ops.append(self.theta_loss)
		ops.append(self.critic_loss)
		ops.append(self.merged)
		
		phi_loss, theta_loss, critic_loss, summary = self.sess.run(ops, feed_dict=feed_dict)
		
		return phi_loss, theta_loss, critic_loss, summary
	
	def train_step_pi(self, states, advantages):
		states = np.reshape(states, (-1, LATENT_DIM))
		advantages = np.reshape(advantages, [-1, 1])
		feed_dict = {self.input_pi_ph: states, self.input_adv: advantages}
		self.sess.run(self.pi_train_step, feed_dict=feed_dict)
	
class game():
	def __init__(self, sess):
		self.sess = sess
		# Game environment
		self.env_name = 'CartPole-v0'
		self.env = gym.make(self.env_name)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.env.reset()
		self.print_game_env()
		st, _, _, _ = self.env.step(self.env.action_space.sample())
		self.net = network(sess)
		self.memory = ReplayBuffer(1000000, 1234)
		self.pi_buffer = ReplayBuffer(10000)
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.print_network()
		self.writer = tf.summary.FileWriter(LOG, sess.graph)
		checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")

	def print_game_env(self):
		print("-----")
		print("environment: {}".format(self.env_name))
		print("action space : {}".format(self.action_space))
		screen = self.env.render(mode='rgb_array')
		print("state space : {}".format(self.observation_space))
		print("observation space : {}".format(screen.shape))
		print("reward range : {}".format(self.env.reward_range))
		print("------\n")
		
	def print_network(self):
		devs= [dev.name for dev in self.net.devs]
		tvars = self.net.net_details
		print("-----")
		print(devs)
		print(" ")
		for var in tvars:
			print(var)
			tensors = tvars[var]
			#tensors = self.sess.run(tensors)
			for v in tensors:
				print(v)
			print(" ")
		print("------\n")
	
	def compute_returns(self, tr):
		returns = [-1.0]
		for sa_pair in tr[::-1]:
			value = 0.1 + 0.9 * returns[-1]
			returns.append(value)
		returns = returns[::-1]
		returns = returns[1:]
		states = [sa[1] for sa in tr]
		for x in range(0, len(states)):
			sas = tr[x]
			sas[3] = returns[x]
			self.memory.add(sas)
		return states, returns
	
	def train_sr(self, step):
		phi_loss, theta_loss, critic_loss, summary = self.net.train_step_sr(self.memory)
		print("step SR {} : phi_loss {}, theta_loss {} critic_loss {} ".format(step, phi_loss, theta_loss, critic_loss))
		self.writer.add_summary(summary, step)
		
	def train_policy(self, step, state, reward):
		self.net.train_step_pi(state, reward)
		print("step PI {}".format(step))
	
	def run_an_episodes(self, epoch, roll_outs, eps):
		STATES, ADVS = [], []
		for roll in range(0, roll_outs):
			done = False
			self.env.reset()
			action = self.env.action_space.sample()
			vec_st, _, _, _ = self.env.step(action)
			img_st = self.env.render(mode='rgb_array')
			img_st, vec_st = pre_process(img_st, vec_st, self.net)
			tr = list()
			while not done:
				at = self.net.get_action(vec_st)
				if eps and (random.random() > 0.6):
					temp = np.zeros((1, 2))
					action = self.env.action_space.sample()
					temp[0][action] = 1.0
					at = temp
				action = np.argmax(at[0])
				temp = np.zeros((1, 2))
				temp[0][action] = 1.0
				at = temp
				vec_st1, rt1, done, info = self.env.step(action)
				img_st1 = self.env.render(mode='rgb_array')
				img_st1, vec_st1 = pre_process(img_st1, vec_st1, self.net)
				sas= [img_st, vec_st, at, rt1, img_st1, vec_st1, done]
				tr.append(sas)
				vec_st, img_st = vec_st1, img_st1
			# memory is updated after returns are computed
			states, returns = self.compute_returns(tr)
			estimates = np.reshape(self.net.get_state_values(states), [-1])
			advantages = returns - estimates
			STATES.extend(states)
			ADVS.extend(advantages)
		print("epoch {} : total return {}".format(epoch, len(STATES)))
		return STATES, ADVS

def main():
	#sess = tf.InteractiveSession()
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	gm = game(sess)
	sr_step, pi_step = 0, 0
	for step in range(10000):
		prob_train = random.random()
		if prob_train > 0.4:
			# eps greedy policy
			STATES, ADVS = gm.run_an_episodes(step, 1, True)
			gm.train_sr(sr_step)
			sr_step += 1
		else:
			# on policy
			STATES, ADVS = gm.run_an_episodes(step, 1, False)
			gm.train_policy(pi_step, STATES, ADVS)
			pi_step += 1
		if step % 1000000 == 0:
			save_path = gm.saver.save(sess, MODEL_PATH + "pretrained.ckpt", global_step=step)
			print("saved to %s" % save_path)

if __name__ == "__main__":
	main()