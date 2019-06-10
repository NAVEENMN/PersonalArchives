import os
import sys
import math
import collections
import tensorflow as tf
import numpy as np
import zipfile
import random

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
log_path = os.path.join(current_path, 'log')
model_path =  os.path.join(current_path, 'saved_model')
data_source = os.path.join(current_path, 'gameofthrones.zip')

data_index = 0

vocabulary_size = 24308
batch_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64
embedding_size = 128
embeddings_shape = [vocabulary_size, embedding_size]
rand_sampled = 2

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    # most_comman orders based on number of occurances and picks n_words words from this
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {}
    # word probs will be sorted by their ids
    word_probablities = []
    # word and is mapping
    for word, num_oc in count:
        dictionary[word] = len(dictionary)
        probablity = float(num_oc)/float(vocabulary_size)
        if probablity < 0:
            probablity = float(0.0)
        word_probablities.append(float(probablity))

    data = []
    unk_count = 0

    # referring to original data
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary, word_probablities


def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # example : the cat sat on the
    # target_word -> sat
    # context words -> the, cat, on, the
    # skip_window -> 2, ==> i.e the cat sat, sat on the
    # num_skips -> 2, ==> number of words randomly sampled from context words
    #                 ==> (sat, cat), (sat, the) i.e two samples
    # span -> 2 * skip_window + 1 ==> if you start at 0, move index to 2 * skip_window + 1
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(int(batch_size / num_skips)):
      context_words = [w for w in range(span) if w != skip_window]
      words_to_use = random.sample(context_words, num_skips)
      for j, context_word in enumerate(words_to_use):
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[context_word]
      if data_index == len(data):
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

class word_to_vec():
    def __init__(self, graph, sess, word_probablities, name):
        self.name = name
        self.sess = sess
        self.graph = graph
        self.word_prob = word_probablities

        with self.graph.as_default():
            # Input data
            with tf.name_scope('inputs'):
                self.input = tf.placeholder(tf.int32, shape=[None], name=self.name+"input")
                self.target = tf.placeholder(tf.int32, shape=[batch_size, 1], name=self.name+"target")
                self.valid_dataset = tf.placeholder(tf.int32, shape=[None], name=self.name+"validate")

            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(tf.random_uniform(embeddings_shape, -1.0, 1.0), name="word_embeddings")

            self.embed = tf.nn.embedding_lookup(self.embeddings, self.input)

            with tf.name_scope('weights'):
                dev = 1.0 / math.sqrt(embedding_size)
                self.nce_weights = tf.Variable(tf.truncated_normal(embeddings_shape, stddev=dev), name="word_weights")

            with tf.name_scope('biases'):
                self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="word_biases")

            self.labels_matrix = tf.reshape(tf.cast(self.target, dtype=tf.int64), [batch_size, 1])

            with tf.name_scope('loss'):
                loss = tf.nn.nce_loss(weights=self.nce_weights,
                                      biases=self.nce_biases,
                                      labels=self.target,
                                      inputs=self.embed,
                                      num_sampled=num_sampled,
                                      num_classes=vocabulary_size)
                self.loss = tf.reduce_mean(loss)

            # Negative sampling.
            self.sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=self.labels_matrix,
                num_true=1,
                num_sampled=rand_sampled,
                unique=True,
                unigrams=self.word_prob,
                range_max=vocabulary_size))

            true_w = tf.nn.embedding_lookup(self.nce_weights, self.target)
            true_b = tf.nn.embedding_lookup(self.nce_biases, self.target)
            sampled_w = tf.nn.embedding_lookup(self.nce_weights, self.sampled_ids)
            sampled_b = tf.nn.embedding_lookup(self.nce_biases, self.sampled_ids)

            true_logits = tf.reduce_sum(tf.multiply(self.embed, true_w), 1) + true_b

            sampled_b_vec = tf.reshape(sampled_b, [rand_sampled])
            # distance between true words and random words
            sampled_logits = tf.matmul(self.embed, sampled_w,transpose_b=True) + sampled_b_vec

            self.nce_loss = self.nce_loss(true_logits, sampled_logits)
            
            with tf.name_scope('optimizer'):
                opt = tf.train.GradientDescentOptimizer(1.0)
                #opt = tf.train.AdamOptimizer(0.01)
                self.train_step = opt.minimize(self.nce_loss)

            # building summaries
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

            self.cos_sim, self.normalized_embeddings = self.cosine_similarity(self.valid_dataset)

    def cosine_similarity(self, valid_dataset):
        with tf.name_scope('cosine_sim'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        return similarity, normalized_embeddings

    # by default we work with normalized embedding
    def get_embedding(self, word_id):
        # example : word_id = dictionary["the"]
        feed_dict = {self.input: np.asarray([word_id])}
        return self.sess.run(self.normalized_embeddings, feed_dict=feed_dict)

    def train_batch(self, batch_inputs, batch_labels, word_probablities):
        feed_dict = {self.input: batch_inputs, self.target: batch_labels}
        run_metadata = tf.RunMetadata()
        ops = [self.train_step, self.loss, self.merged]
        _, loss_val, summary = self.sess.run(ops, feed_dict=feed_dict, run_metadata=run_metadata)
        print(self.sess.run(self.sampled_ids, feed_dict=feed_dict))
        variables_names = [v.name for v in tf.trainable_variables()]
        print(variables_names)
        return loss_val, summary

def main():

    vocabulary = read_data(data_source)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        data, count, dictionary, reverse_dictionary, word_probablities = build_dataset(vocabulary, vocabulary_size)
        print("total", len(count))
        del vocabulary

        wvec = word_to_vec(graph, sess, word_probablities, "word_2_vec_")
        writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()  # init the graph

        # testing data
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
        batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

        num_steps = 30000
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
            loss, summary = wvec.train_batch(batch_inputs, batch_labels, word_probablities)
            average_loss += loss

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            writer.add_summary(summary, step)

        with open(log_path + '/metadata.tsv', 'w') as f:
            for i in range(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        saver.save(sess, os.path.join(model_path, 'model.ckpt'))

    writer.close()

if __name__ == "__main__":
    main()
