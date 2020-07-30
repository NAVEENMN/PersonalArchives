import tensorflow as tf

TENSORBOARD_PATH = "Tensorboard\\"
MODEL_PATH = "models\\"
# Image Parameters
image_in = {"width": 28, "height": 28, "channel": 1, "n_pixels":28*28*1,"shape":[28, 28, 1],"batch_size":50}
image_out = {"width": 28, "height": 28, "channel": 1, "n_pixels":28*28*1,"shape":[28, 28, 1],"batch_size":50}


def tensorboard_summary(summary, gen):
	for key in summary:
		tf.summary.scalar(key, summary[key])
	images = tf.reshape(gen, [image_in["batch_size"], image_in["width"], image_in["height"], image_in["channel"]])
	tf.summary.image('Generated_images', images, 10)
	merged = tf.summary.merge_all()
	return merged

def model_log(sess):
	writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)
	print(TENSORBOARD_PATH)
	saver = tf.train.Saver()
	return writer, saver

def model_op(sess, saver, g_step=0, op="load"):
	if op == "load":
		checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
	if op == "save":
		save_path = saver.save(sess, MODEL_PATH + "\\pretrained_gan.ckpt", global_step=g_step)
		print("saved to %s" % save_path)


def print_summary(avg_d_loss, avg_g_loss, avg_vae_loss, mini_batches, epoch):
	avg_d_loss = avg_d_loss / mini_batches
	avg_g_loss = avg_g_loss / mini_batches
	avg_vae_loss = avg_vae_loss / mini_batches
	print("Epoch:", '%04d' % (epoch), \
		  "avg_d_loss=", "{:.6f}".format(avg_d_loss), \
		  "avg_g_loss=", "{:.6f}".format(avg_g_loss), \
		  "avg_vae_loss=", "{:.6f}".format(avg_vae_loss))
