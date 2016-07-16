import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Word2Vec, Options
from six.moves import xrange
import os
import time

class MyWord2Vec(Word2Vec):
    # override
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()

    # override save_vocab
    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]).encode('utf-8'),
                    opts.vocab_counts[i]))

def main():
    # set word2vec options
    options = Options()
    options.save_path = "tf/"
    options.train_data = "test.txt"
    options.batch_szie = 5000
    options.window_size = 4
    options.subsample = 0
    options.epochs_to_train = 5
    options.concurrent_steps = 4


    with tf.Graph().as_default():
        with tf.Session() as session:
            with tf.device("/cpu:0"):
                model = MyWord2Vec(options, session)
                for _ in xrange(options.epochs_to_train):
                    model.train()  # Process one epoch
                    #model.eval() # Eval analogies.

                model.saver.save(session,
                     os.path.join(options.save_path, "model.ckpt"),
                     global_step=model.global_step)

if __name__ == '__main__':
    start = time.time()
    main()
    spend = time.time() - start
    print ""
    print "Spend time : %d s" % (spend)
