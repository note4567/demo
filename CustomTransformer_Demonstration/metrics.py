import tensorflow as tf  # NOQA # type: ignore


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size) -> tuple:
    with tf.name_scope("loss"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        with tf.name_scope("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.cast(
                vocab_size - 1, dtype=tf.float32
            )

            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence,
            )

            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets
            )

            normalizing_constant = -(
                confidence * tf.math.log(confidence)
                + tf.cast(vocab_size - 1, dtype=tf.float32)
                * low_confidence
                * tf.math.log(low_confidence + 1e-20)
            )
            xentropy -= normalizing_constant

        weights = tf.cast(tf.not_equal(labels, 0), dtype=tf.float32)
        return xentropy * weights, weights


def padded_accuracy(logits, labels) -> tuple:
    with tf.name_scope("padded_accuracy"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), dtype=tf.float32)
        outputs = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)

        padded_labels = tf.cast(labels, dtype=tf.int32)
        return (
            tf.cast(tf.equal(outputs, padded_labels), dtype=tf.float32),
            weights,
        )  # NOQA


def _pad_tensors_to_same_length(x, y) -> tuple:
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]
        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])

        return x, y
