from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')


"""
gcc compute-accuracy_euclidean.c -o compute-accuracy_euclidean
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
python word2vec_optimized.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import threading
import time
import pickle

import subprocess
import operator

import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import math

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

## BIG DATASET
flags.DEFINE_string("save_path", "./", "Directory to write the model.")
flags.DEFINE_string("train_data", "../data/enwiki-20170720-pages-articles_LOWER_AND_UNICODE_without_punctuations_with_freebase_entity_labels.txt", "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("eval_data", "/home/paolo/modified_w2v/pTransE/pTransE-master/ptranse_only_text_batch_sizes/OLD/data/questions-words_lower.txt", "Analogy questions. See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 100, "The embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 3, "Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", float(sys.argv[1]), "Initial learning rate.") #0.025
flags.DEFINE_integer("num_neg_samples", 10, "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500, "Numbers of training examples each step processes (no minibatching).")
flags.DEFINE_integer("concurrent_steps", 16, "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("min_count", 45, "The minimum number of word occurrences for it to be included in the vocabulary.") #0 #450 #70
flags.DEFINE_float("subsample", 1e-3, "Subsample threshold for word occurrence. Words that appear with higher frequency will be randomly down-sampled. Set to 0 to disable.")
flags.DEFINE_boolean("interactive", False, "If true, enters an IPython interactive session to play with the trained model. E.g., try model.analogy(b'france', b'paris', b'russia') and model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_string("train_data_kg", "../data/freebase_ids_label_without_punctuation_without_duplicates_train_top200kEntities", "Training data of the KG.")

FLAGS = flags.FLAGS

class Options(object):
    """Options used by our word2vec model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # Training options.

        # The training text file.
        self.train_data = FLAGS.train_data

        # The training KG file.
        self.train_data_kg = FLAGS.train_data_kg

        # Number of negative samples per example.
        self.num_samples = FLAGS.num_neg_samples

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        # self.percentage_data = FLAGS.percentage_data
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the vocabulary.
        self.min_count = FLAGS.min_count

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # The text file for eval.
        self.eval_data = FLAGS.eval_data


class Word2Vec(object):

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._entity2id = {}
        self._relation2id = {}
        self._id2word = []
        self._id2entity = []
        self._id2relation = []
        self.build_graph()
        self.build_eval_graph()

    def read_analogies(self):
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", self._options.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)

    def build_graph(self):
        opts = self._options
        print("opts.batch_size: " + str(opts.batch_size))

        (words, counts, words_per_epoch, current_epoch, total_words_processed, examples, labels, heads, relations, tails, vocab_entities, vocab_relations, counts_ent, counts_rel, facts_per_epoch, total_facts_processed) = word2vec.skipgram_word2vec(filename=opts.train_data, batch_size=opts.batch_size, window_size=opts.window_size, min_count=opts.min_count, subsample=opts.subsample, filename_kg=opts.train_data_kg)


        (opts.vocab_words, opts.vocab_counts, opts.words_per_epoch, opts.vocab_entities, opts.vocab_relations, opts.ent_counts, opts.rel_counts, opts.facts_per_epoch) = self._session.run([words, counts, words_per_epoch, vocab_entities, vocab_relations, counts_ent, counts_rel, facts_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        opts.vocab_ent_size = len(opts.vocab_entities)
        opts.vocab_rel_size = len(opts.vocab_relations)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)
        print("Facts per epoch: ", opts.facts_per_epoch)

        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
            # print(str(w) + ": " + str(i))

        self._id2entity = opts.vocab_entities
        for i, w in enumerate(self._id2entity):
            self._entity2id[w] = i
            # print(str(w) + ": " + str(i))

        self._id2relation = opts.vocab_relations
        for i, w in enumerate(self._id2relation):
            self._relation2id[w] = i

        print("Word size: " + str(opts.vocab_size) + ", Emb size: " + str(opts.vocab_ent_size) + ", Rel size: " + str(opts.vocab_rel_size))
        w_in = tf.Variable(tf.random_uniform([opts.vocab_size, opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim), name="w_in")

        # w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")
        w_out = tf.Variable(tf.random_uniform([opts.vocab_size, opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim), name="w_out")

        w_in_ent = tf.Variable(tf.random_uniform([opts.vocab_ent_size, opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim), name="w_in_ent")

        w_in_rel = tf.Variable(tf.random_uniform([opts.vocab_rel_size, opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim), name="w_in_rel")

        global_step = tf.Variable(0, name="global_step")


        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        print("opts.words_per_epoch: " + str(opts.words_per_epoch))
        print("opts.epochs_to_train: " + str(opts.epochs_to_train))
        print("words_to_train: " + str(words_to_train))
        print("total_words_processed.eval(): " + str(total_words_processed.eval()))
        lr = opts.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)
        # lr = opts.learning_rate * tf.maximum(1.0, 1.0)

        inc = global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            train = word2vec.neg_train_word2vec(w_in, examples, labels, lr, w_in_rel, heads, relations, tails, w_in_ent, w_out, vocab_count=opts.vocab_counts.tolist(), num_negative_samples=opts.num_samples, ent_count=opts.ent_counts.tolist(), rel_count=opts.rel_counts.tolist())

        self._w_in = w_in
        self._examples = examples
        self._labels = labels
        self._lr = lr
        self.w_in_ent = w_in_ent
        self.w_in_rel = w_in_rel
        self.heads = heads
        self.relations = relations
        self.tails = tails
        self._train = train
        self.global_step = global_step
        self._epoch = current_epoch
        self._words = total_words_processed
        self._facts = total_facts_processed

    def save_vocab(self):
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                f.write("%s %d\n" % (vocab_word, opts.vocab_counts[i]))

    def save_emb(self):
        opts = self._options
        all_embeddings_ent = self.w_in_ent.eval()
        np.savetxt(os.path.join(opts.save_path, "entity2vec.bern"), all_embeddings_ent)
        all_embeddings_rel = self.w_in_rel.eval()
        np.savetxt(os.path.join(opts.save_path, "relation2vec.bern"), all_embeddings_rel)

        sorted_entity2id = sorted(self._entity2id.items(), key=operator.itemgetter(1))
        f = open(os.path.join(opts.save_path, "entity2id.txt"), 'w')
        for el in sorted_entity2id:
            f.write(str(el[0]) + "\t" + str(el[1]) + "\n")

        sorted_relation2id = sorted(self._relation2id.items(), key=operator.itemgetter(1))
        f = open(os.path.join(opts.save_path, "relation2id.txt"), 'w')
        for el in sorted_relation2id:
            f.write(str(el[0]) + "\t" + str(el[1]) + "\n")

        all_embeddings = self._w_in.eval()
        with open(os.path.join(opts.save_path, "emb_text.bin"), "w") as f:
            f.write(str(len(opts.vocab_words)) + " " + str(opts.emb_dim))
            print("opts.vocab_size: " + str(opts.vocab_size))
            for i in range(len(all_embeddings)):
                word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                bin_emb = all_embeddings[i].tobytes()
                f.write('\n'+str(word) + ' ' + bin_emb)
                if i%1000 == 0:
                    print(i)

    def save_emb_backup(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        opts = self._options
        all_embeddings_ent = self.w_in_ent.eval()
        np.savetxt(os.path.join(model_path, "entity2vec.bern"), all_embeddings_ent)
        all_embeddings_rel = self.w_in_rel.eval()
        np.savetxt(os.path.join(model_path, "relation2vec.bern"), all_embeddings_rel)

        sorted_entity2id = sorted(self._entity2id.items(), key=operator.itemgetter(1))
        f = open(os.path.join(model_path, "entity2id.txt"), 'w')
        for el in sorted_entity2id:
            f.write(str(el[0]) + "\t" + str(el[1]) + "\n")

        sorted_relation2id = sorted(self._relation2id.items(), key=operator.itemgetter(1))
        f = open(os.path.join(model_path, "relation2id.txt"), 'w')
        for el in sorted_relation2id:
            f.write(str(el[0]) + "\t" + str(el[1]) + "\n")

        all_embeddings = self._w_in.eval()
        with open(os.path.join(model_path, "emb_text.bin"), "w") as f:
            f.write(str(len(opts.vocab_words)) + " " + str(opts.emb_dim))
            print("opts.vocab_size: " + str(opts.vocab_size))
            for i in range(len(all_embeddings)):
                word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                bin_emb = all_embeddings[i].tobytes()
                f.write('\n'+str(word) + ' ' + bin_emb)
                if i%1000 == 0:
                    print(i)

    def build_eval_graph(self):
        opts = self._options

        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)

        nemb = tf.nn.l2_normalize(self._w_in, 1)

        a_emb = tf.gather(nemb, analogy_a)
        b_emb = tf.gather(nemb, analogy_b)
        c_emb = tf.gather(nemb, analogy_c)

        target = c_emb + (b_emb - a_emb)

        dist = tf.matmul(target, nemb, transpose_b=True)

        _, pred_idx = tf.nn.top_k(dist, 4)

        nearby_word = tf.placeholder(dtype=tf.int32)
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, opts.vocab_size))

        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break

    def train(self):
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time = initial_words, time.time()
        while True:
            time.sleep(5)  # Reports our progress once a while.
            (epoch, step, words, lr) = self._session.run([self._epoch, self.global_step, self._words, self._lr])
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
            print("Epoch %4d Step %8d: lr = %5.4f words/sec = %8.0f\r" % (epoch, step, lr, rate), end="")
            sys.stdout.flush()
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()

    def _predict(self, analogy):
        idx, = self._session.run([self._analogy_pred_idx], {self._analogy_a: analogy[:, 0], self._analogy_b: analogy[:, 1], self._analogy_c: analogy[:, 2]})
        return idx

    def eval(self):
        correct = 0

        try:
            total = self._analogy_questions.shape[0]
        except AttributeError as e:
            raise AttributeError("Need to read analogy questions.")

        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
						# Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        continue
                    else:
                        break
        print()
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))

    def analogy(self, w0, w1, w2):
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                print(c)
                break
        print("unknown")

    def nearby(self, words, num=20):
        ids = np.array([self._word2id.get(x, 0) for x in words])
        vals, idx = self._session.run([self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in xrange(len(words)):
            print("\n%s\n=====================================" % (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self._id2word[neighbor], distance))

def get_numb_of_facts_kg(train_data_kg):
    with open(train_data_kg) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_numb_of_words_text(train_data):
    with open(train_data) as f:
        words = [word for line in f for word in line.split()]
        return len(words)

def _start_shell(local_ns=None):
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

def main(_):
    start_time = time.time()
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Word2Vec(opts, session)
            mean_raw_left = []
            hit_raw_left = []
            mean_filtered_left = []
            hit_filtered_left = []
            mean_raw_right = []
            hit_raw_right = []
            mean_filtered_right = []
            hit_filtered_right = []
        for _ in xrange(opts.epochs_to_train):
            print ("\nepoch: " + str(_))
            sys.stdout.flush()
            model.train()
            model._w_in = tf.nn.l2_normalize(model._w_in, 1)
            model.w_in_ent = tf.nn.l2_normalize(model.w_in_ent, 1)
            model.w_in_rel = tf.nn.l2_normalize(model.w_in_rel, 1)

            #save at the end of each epoch
            model_path = "./backup/" + str(_)
            model.save_emb_backup(model_path)

            # model.save_emb()
            #
            # ##TransE evaluation euclidean
            # print ("\nTransE evaluation euclidean")
            # proc = subprocess.Popen(["./Test_TransE_euclidean", "bern"], stdout=subprocess.PIPE)
            # output = proc.stdout.read()
            # print ("\n\noutput: " + str(output) + "\n\n")
            # sys.stdout.flush()
            #
            # ##Analogy euclidian norm
            # print ("\nAnalogy euclidian norm")
            # proc = subprocess.Popen(["./compute-accuracy_euclidean emb_text.bin < /home/ubuntu/data/questions-words_lower.txt"], stdout=subprocess.PIPE, shell=True)
            # output = proc.stdout.read()
            # print ("\n\noutput: " + str(output) + "\n\n")
            # sys.stdout.flush()

        if FLAGS.interactive:
            _start_shell(locals())

    print("--- %s Total execution time optimized---" % (time.time() - start_time))


if __name__ == "__main__":
    tf.app.run()
