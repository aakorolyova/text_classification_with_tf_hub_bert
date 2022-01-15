import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from input_example import InputExample, PaddingInputExample


def create_tokenizer_from_hub_module(bert_path, session):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = session.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        return input_ids, input_mask, segment_ids

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_pair_example(tokenizer, example, max_seq_length_a=256, max_seq_length_b=256):
    """Converts an `InputExample` for a sequence labelling task (text_b not None)
    into a single `InputFeatures`.
    """

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * (max_seq_length_a + max_seq_length_b)
        input_mask = [0] * (max_seq_length_a + max_seq_length_b)
        segment_ids = [0] * (max_seq_length_a + max_seq_length_b)
        return input_ids, input_mask, segment_ids

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length_a - 2:
        tokens_a = tokens_a[0: (max_seq_length_a - 2)]

    tokens_b = tokenizer.tokenize(example.text_b)
    if len(tokens_b) > max_seq_length_b:
        tokens_b = tokens_b[0: max_seq_length_b]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence a length.
    while len(input_ids) < max_seq_length_a - 1:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids.append(0)
    input_mask.append(1)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(0)

    input_ids.extend(tokenizer.convert_tokens_to_ids(tokens_b))
    input_mask.extend([1] * len(tokens_b))

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length_a + max_seq_length_b:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    assert len(input_ids) == max_seq_length_a + max_seq_length_b
    assert len(input_mask) == max_seq_length_a + max_seq_length_b
    assert len(segment_ids) == max_seq_length_a + max_seq_length_b

    return input_ids, input_mask, segment_ids


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids = [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids)
    )


def convert_pair_examples_to_features(tokenizer, examples, max_seq_length_a=256, max_seq_length_b=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids = [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id = convert_pair_example(
            tokenizer, example, max_seq_length_a, max_seq_length_b
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids)
    )

def convert_text_to_examples(texts):
    """Create InputExamples"""
    InputExamples = []
    for text in texts:
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None)
        )
    return InputExamples


def convert_pair_text_to_examples(texts_a, texts_b):
    """Create InputExamples"""
    InputExamples = []
    for text_a, text_b in zip(texts_a, texts_b):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text_a), text_b=" ".join(text_b))
        )
    return InputExamples


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)