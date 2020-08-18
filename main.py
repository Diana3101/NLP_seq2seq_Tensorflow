import random


def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    """Generates pairs of equations and solutions to them.

       Each equation has a form of two integers with an operator in between.
       Each solution is an integer with the result of the operaion.

        allowed_operators: list of strings, allowed operators.
        dataset_size: an integer, number of equations to be generated.
        min_value: an integer, min value of each operand.
        max_value: an integer, max value of each operand.

        result: a list of tuples of strings (equation, solution).
    """
    sample = []
    for _ in range(dataset_size):
        x = random.randint(min_value, max_value)
        y = random.randint(min_value, max_value)
        equation = str(x) + random.choice(allowed_operators) + str(y)
        sample.append((equation, str(eval(equation))))
    return sample


def test_generate_equations():
    allowed_operators = ['+', '-']
    dataset_size = 10
    for (input_, output_) in generate_equations(allowed_operators, dataset_size, 0, 100):
        if not (type(input_) is str and type(output_) is str):
            return "Both parts should be strings."
        if eval(input_) != int(output_):
            return "The (equation: {!r}, solution: {!r}) pair is incorrect.".format(input_, output_)
    return "Tests passed."


print(test_generate_equations())
# allowed_operators = ['+', '-']
# print(generate_equations(allowed_operators, 10, 0, 100))

from sklearn.model_selection import train_test_split

allowed_operators = ['+', '-']
dataset_size = 100000
data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=9999)

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

word2id = {symbol: i for i, symbol in enumerate('#^$+-1234567890')}
id2word = {i: symbol for symbol, i in word2id.items()}

# print(id2word)

start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'


def sentence_to_ids(sentence, word2id, padded_len):
    """ Converts a sequence of symbols to a padded sequence of their ids.

      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.

      result: a tuple of (a list of ids, an actual length of sentence).
    """
    sent_ids = []
    for symbol in sentence:
        if len(sent_ids) < padded_len - 1:
            for i in word2id.values():
                if word2id[symbol] == i:
                    sent_ids.append(i)
    sent_ids.append(word2id['$'])
    sent_len = len(sent_ids)
    while padded_len - len(sent_ids) >= 1:
        sent_ids.append(word2id['#'])
    return sent_ids, sent_len


def test_sentence_to_ids():
    sentences = [("123+123", 7), ("123+123", 8), ("123+123", 10)]
    expected_output = [([5, 6, 7, 3, 5, 6, 2], 7),
                       ([5, 6, 7, 3, 5, 6, 7, 2], 8),
                       ([5, 6, 7, 3, 5, 6, 7, 2, 0, 0], 8)]
    for (sentence, padded_len), (sentence_ids, expected_length) in zip(sentences, expected_output):
        output, length = sentence_to_ids(sentence, word2id, padded_len)
        if output != sentence_ids:
            return ("Convertion of '{}' for padded_len={} to {} is incorrect.".format(
                sentence, padded_len, output))
        if length != expected_length:
            return ("Convertion of '{}' for padded_len={} has incorrect actual length {}.".format(
                sentence, padded_len, length))
    return "Tests passed."


print(test_sentence_to_ids())


def ids_to_sentence(ids, id2word):
    """ Converts a sequence of ids to a sequence of symbols.

          ids: a list, indices for the padded sequence.
          id2word:  a dict, a mapping from ids to original symbols.

          result: a list of symbols.
    """

    return [id2word[i] for i in ids]


def batch_to_ids(sentences, word2id, max_len):
    """Prepares batches of indices.

       Sequences are padded to match the longest sequence in the batch,
       if it's longer than max_len, then max_len is used instead.

        sentences: a list of strings, original sequences.
        word2id: a dict, a mapping from original symbols to ids.
        max_len: an integer, max len of sequences allowed.

        result: a list of lists of ids, a list of actual lengths.
    """

    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len


def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y


sentences = train_set[0]
ids, sent_lens = batch_to_ids(sentences, word2id, max_len=10)
print('Input:', sentences)
print('Ids: {}\nSentences lengths: {}'.format(ids, sent_lens))

import tensorflow as tf


class Seq2SeqModel(object):
    pass


def declare_placeholders(self):
    """Specifies placeholders for the model."""

    # Placeholders for input and its actual lengths.
    self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
    self.input_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_batch_lengths')

    # Placeholders for groundtruth and its actual lengths.
    self.ground_truth = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ground_truth')
    self.ground_truth_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='ground_truth_lengths')

    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
    self.learning_rate_ph = tf.placeholder_with_default(tf.cast(0.01, tf.float32), shape=[])


Seq2SeqModel.__declare_placeholders = classmethod(declare_placeholders)


def create_embeddings(self, vocab_size, embeddings_size):
    """Specifies embeddings layer and embeds an input batch."""

    random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
    self.embeddings = tf.Variable(random_initializer, dtype=tf.float32, name='embeddings')

    # Perform embeddings lookup for self.input_batch.
    self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_batch)


Seq2SeqModel.__create_embeddings = classmethod(create_embeddings)


def build_encoder(self, hidden_size):
    """Specifies encoder architecture and computes its output."""

    # Create GRUCell with dropout.
    encoder_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.GRUCell(hidden_size),
        output_keep_prob=self.dropout_ph
    )

    # Create RNN with the predefined cell.
    _, self.final_encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.input_batch_embedded, dtype=tf.float32)


Seq2SeqModel.__build_encoder = classmethod(build_encoder)


def build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
    """Specifies decoder architecture and computes the output.

        Uses different helpers:
          - for train: feeding ground truth
          - for inference: feeding generated output

        As a result, self.train_outputs and self.infer_outputs are created.
        Each of them contains two fields:
          rnn_output (predicted logits)
          sample_id (predictions).

    """

    # Use start symbols as the decoder inputs at the first time step.
    batch_size = tf.shape(self.input_batch)[0]
    start_tokens = tf.fill([batch_size], start_symbol_id)
    ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

    # Use the embedding layer defined before to lookup embedings for ground_truth_as_input.
    self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings, ground_truth_as_input)

    # Create TrainingHelper for the train stage.
    train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded,
                                                     self.ground_truth_lengths)

    # Create GreedyEmbeddingHelper for the inference stage.
    # You should provide the embedding layer, start_tokens and index of the end symbol.
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_symbol_id)

    def decode(helper, scope, reuse=None):
        """Creates decoder and return the results of the decoding with a given helper."""

        with tf.variable_scope(scope, reuse=reuse):
            # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(hidden_size, reuse=reuse),
                output_keep_prob=self.dropout_ph
            )

            # Create a projection wrapper.
            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)

            # Create BasicDecoder, pass the defined cell, a helper, and initial state.
            # The initial state should be equal to the final state of the encoder!
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, self.final_encoder_state)

            # The first returning argument of dynamic_decode contains two fields:
            #   rnn_output (predicted logits)
            #   sample_id (predictions)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter,
                                                              output_time_major=False, impute_finished=True)

            return outputs

    self.train_outputs = decode(train_helper, 'decode')
    self.infer_outputs = decode(infer_helper, 'decode', reuse=True)


Seq2SeqModel.__build_decoder = classmethod(build_decoder)


def compute_loss(self):
    """Computes sequence loss (masked cross-entopy loss with logits)."""

    weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)

    self.loss = tf.contrib.seq2seq.sequence_loss(self.train_outputs.rnn_output, self.ground_truth, weights)


Seq2SeqModel.__compute_loss = classmethod(compute_loss)


def perform_optimization(self):
    """Specifies train_op that optimizes self.loss."""

    self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=tf.train.get_global_step(),
                                                    learning_rate=self.learning_rate_ph, optimizer='Adam')


Seq2SeqModel.__perform_optimization = classmethod(perform_optimization)


def init_model(self, vocab_size, embeddings_size, hidden_size,
               max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):
    self.__declare_placeholders()
    self.__create_embeddings(vocab_size, embeddings_size)
    self.__build_encoder(hidden_size)
    self.__build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

    # Compute loss and back-propagate.
    self.__compute_loss()
    self.__perform_optimization()

    # Get predictions for evaluation.
    self.train_predictions = self.train_outputs.sample_id
    self.infer_predictions = self.infer_outputs.sample_id


Seq2SeqModel.__init__ = classmethod(init_model)


def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, learning_rate, dropout_keep_probability):
    feed_dict = {
        self.input_batch: X,
        self.input_batch_lengths: X_seq_len,
        self.ground_truth: Y,
        self.ground_truth_lengths: Y_seq_len,
        self.learning_rate_ph: learning_rate,
        self.dropout_ph: dropout_keep_probability
    }
    pred, loss, _ = session.run([
        self.train_predictions,
        self.loss,
        self.train_op], feed_dict=feed_dict)
    return pred, loss


Seq2SeqModel.train_on_batch = classmethod(train_on_batch)


def predict_for_batch(self, session, X, X_seq_len):
    feed_dict = {
        self.input_batch: X,
        self.input_batch_lengths: X_seq_len
    }
    pred = session.run([
        self.infer_predictions
    ], feed_dict=feed_dict)[0]
    return pred


def predict_for_batch_with_loss(self, session, X, X_seq_len, Y, Y_seq_len):
    feed_dict = {
        self.input_batch: X,
        self.input_batch_lengths: X_seq_len,
        self.ground_truth: Y,
        self.ground_truth_lengths: Y_seq_len
    }
    pred, loss = session.run([
        self.infer_predictions,
        self.loss,
    ], feed_dict=feed_dict)
    return pred, loss


Seq2SeqModel.predict_for_batch = classmethod(predict_for_batch)
Seq2SeqModel.predict_for_batch_with_loss = classmethod(predict_for_batch_with_loss)

tf.reset_default_graph()

model = Seq2SeqModel(vocab_size=len(word2id), embeddings_size=20, max_iter=7,
                      hidden_size=512, start_symbol_id=word2id['^'], end_symbol_id=word2id['$'], padding_symbol_id=word2id['#'])

batch_size = 128
n_epochs = 10
learning_rate = 0.001
dropout_keep_probability = 0.5  # 0.1-1.0
max_len = 20

n_step = int(len(train_set) / batch_size)

session = tf.Session()
session.run(tf.global_variables_initializer())

invalid_number_prediction_counts = []
all_model_predictions = []
all_ground_truth = []

print('Start training... \n')
for epoch in range(n_epochs):
    random.shuffle(train_set)
    random.shuffle(test_set)

    print('Train: epoch', epoch + 1)
    for n_iter, (X_batch, Y_batch) in enumerate(generate_batches(train_set, batch_size=batch_size)):
        X_batch_ids, X_batch_len = batch_to_ids(X_batch, word2id, max_len)
        Y_batch_ids, Y_batch_len = batch_to_ids(Y_batch, word2id, max_len)
        # prepare the data (X_batch and Y_batch) for training
        # using function batch_to_ids
        predictions, loss = model.train_on_batch(session, X_batch_ids, X_batch_len, Y_batch_ids, Y_batch_len,
                                                 learning_rate, dropout_keep_probability)

        if n_iter % 200 == 0:
            print("Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch + 1, n_epochs, n_iter + 1, n_step, loss))

    X_sent, Y_sent = next(generate_batches(test_set, batch_size=batch_size))
    X, X_sent_len = batch_to_ids(X_sent, word2id, max_len)
    Y, Y_sent_len = batch_to_ids(Y_sent, word2id, max_len)
    # prepare test data (X_sent and Y_sent) for predicting
    # quality and computing value of the loss function
    # using function batch_to_ids

    predictions, loss = model.train_on_batch(session, X, X_sent_len, Y, Y_sent_len,
                                             learning_rate, dropout_keep_probability)
    print('Test: epoch', epoch + 1, 'loss:', loss, )
    for x, y, p in list(zip(X, Y, predictions))[:3]:
        print('X:', ''.join(ids_to_sentence(x, id2word)))
        print('Y:', ''.join(ids_to_sentence(y, id2word)))
        print('O:', ''.join(ids_to_sentence(p, id2word)))
        print('')

    model_predictions = []
    ground_truth = []
    invalid_number_prediction_count = 0
    # For the whole test set calculate ground-truth values (as integer numbers)
    # and prediction values (also as integers) to calculate metrics.
    # If generated by model number is not correct (e.g. '1-1'),
    # increase invalid_number_prediction_count and don't append this and corresponding
    # ground-truth value to the arrays.
    for X_batch, Y_batch in generate_batches(test_set, batch_size=batch_size):
        X_batch_ids, X_batch_len = batch_to_ids(X_batch, word2id, max_len)
        preds = model.predict_for_batch(session, X_batch_ids, X_batch_len)
        for i in range(len(preds)):
            model_predictions.append(int(''.join(ids_to_sentence(preds[i], id2word)).split('$')[0]))
            ground_truth.append(int(Y_batch[i]))
    all_model_predictions.append(model_predictions)
    all_ground_truth.append(ground_truth)
    invalid_number_prediction_counts.append(invalid_number_prediction_count)

print('\n...training finished.')

from sklearn.metrics import mean_absolute_error

for i, (gts, predictions, invalid_number_prediction_count) in enumerate(zip(all_ground_truth,
                                                                            all_model_predictions,
                                                                            invalid_number_prediction_counts), 1):
    mae = mean_absolute_error(gts, predictions)
    print("Epoch: %i, MAE: %f, Invalid numbers: %i" % (i, mae, invalid_number_prediction_count))
