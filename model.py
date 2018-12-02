import tensorflow as tf
import numpy as np
import os
import metrics
from algorithms import *


class Model(object):

    class AccuracyType:
        FULL_SEQUENCE, EDIT_DISTANCE = range(2)

    def __init__(self, convolutional_network, seq2se2_param, num_classes, data_format, accuracy_type, hyper_params):

        self.convolutional_network = convolutional_network
        self.seq2se2_param = seq2se2_param
        self.num_classes = num_classes
        self.data_format = data_format
        self.accuracy_type = accuracy_type
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["image"]

        feature_maps = self.convolutional_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors = flatten_images(feature_maps, self.data_format)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.seq2seq_param.lstm_units,
            use_peepholes=True
        )

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.seq2se2_param.attention_units,
            memory=feature_vectors
        )

        ''' Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the `normalizer`.
        - Step 5: Calculate the context vector as the inner product 
                  between the alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output and context 
                  through the attention layer (a linear layer with `attention_layer_size` outputs).
        '''
        attention_lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=lstm_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.seq2se2_param.attention_layer_size,
            cell_input_fn=lambda inputs, attention: tf.layers.dense(
                inputs=tf.concat([inputs, attention], axis=-1),
                units=self.seq2se2_param.attention_layer_size
            ),
            output_attention=True
        )

        batch_size, time_step = tf.unstack(tf.shape(feature_vectors)[:-1], axis=0)
        start_token = end_token = -1

        if mode == tf.estimator.ModeKeys.TRAIN:

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.one_hot(
                    indices=tf.concat([tf.tile([start_token], [batch_size])[:, tf.newaxis], labels[:, :-1]], axis=1),
                    depth=self.num_classes
                ),
                sequence_length=tf.tile([time_step], [batch_size]),
                time_major=False
            )

        else:

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=lambda labels: tf.one_hot(
                    indices=labels,
                    depth=self.num_classes
                ),
                start_tokens=tf.tile([start_token], [batch_size]),
                end_token=end_token
            )

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attention_lstm_cell,
            helper=helper,
            initial_state=attention_lstm_cell.zero_state(batch_size, tf.float32),
            output_layer=tf.layers.Dense(self.num_classes)
        )

        outputs, state, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=time_step,
            parallel_iterations=os.cpu_count(),
            swap_memory=False,
        )

        logits = outputs.rnn_output

        loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=labels,
            weights=tf.sequence_mask(sequence_lengths, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True
        )

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                train_op = tf.train.AdamOptimizer().minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            accuracy_functions = {
                Model.AccuracyType.FULL_SEQUENCE: metrics.full_sequence_accuracy,
                Model.AccuracyType.EDIT_DISTANCE: metrics.edit_distance_accuracy,
            }

            accuracy = accuracy_functions[self.accuracy_type](
                logits=logits,
                labels=labels,
                time_major=False
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(accuracy=accuracy)
            )
