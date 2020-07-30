from models import drive
from utils import util
import tensorflow as tf

MODEL_PATH = "saved_models\\"
LOG = "Tensorboard\\"

class control():
	def __init__(self):
		self.sess = sess = tf.InteractiveSession()
		self.drive_ = drive.drive()
		self.data = util.batch_generator()
		self.saver = tf.train.Saver()
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(LOG, sess.graph)
		self.sess.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
	def get_action(self, image):
		response = self.drive_.feed_forward(image)
		print(response)
	def get_state(self):
		state = util.get_state()
		return state
	def visulize(self):
		self.data.visulize_an_episode()
	def train_drive(self):
		batch_id = 0
		for data in self.data.next_batch():
			feed_dict = self.drive_.train(sess=self.sess,data=data, batch_id=batch_id)
			batch_id += 1
			if batch_id % 500 == 0:
				save_path = self.saver.save(self.sess, MODEL_PATH + "\\pretrained.ckpt", global_step=batch_id)
				print("saved to: ", save_path)
			if batch_id % 10 == 0:
				summary = util.merge_summary(self.sess, self.merged, feed_dict)
				self.writer.add_summary(summary, batch_id)
		
def main():
	print("--flying--")
	c = control()
	#c.visulize()
	c.train_drive()
	
if __name__ == "__main__":
	main()