import numpy as np
import tensorflow as tf
import io
import functools

from skimage.filters import threshold_local

from protos import input_reader_pb2
from data_decoders import tf_example_decoder



def _load_image_into_bytes(image):
    with io.BytesIO() as imByteArry:
        image.save(imByteArry, format='jpeg')
        return imByteArry.getvalue()

# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)

def read_dataset(file_read_func, input_files, config):
    filenames = tf.gfile.Glob(input_files)
    num_readers= config.num_readers
    if num_readers > len(filenames):
        num_readers = len(filenames)
        tf.logging.warning('')

    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if config.shuffle:
        filename_dataset = filename_dataset.shuffle(
            config.filenames_shuffle_buffer_size)
    elif num_readers > 1:
        tf.logging.warning('')
    filename_dataset = filename_dataset.repeat(config.num_epochs or None)
    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_func,
            cycle_length=num_readers or 1,
            # block_length=config.read_block_length or 1,
            sloppy=config.shuffle
    ))

    # if config.shuffle:
    #     records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)

    return records_dataset

def build(input_reader_config, batch_size=None, transform_input_data_fn=None):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError()

    if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
        config = input_reader_config.tf_record_input_reader
        if not config.input_path:
            raise ValueError('')

        decoder = tf_example_decoder.TfExampleDecoder()

        def process_fn(value):
            processed_tensors = decoder.decode(value)
            if transform_input_data_fn is not None:
                processed_tensors = transform_input_data_fn(processed_tensors)
            return processed_tensors

        # Read input_path to string datatset
        dataset = read_dataset(
            functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
            config.input_path[:],
            input_reader_config)


        if batch_size:
            num_parallel_calls = batch_size * input_reader_config.num_parallel_batches
        else:
            num_parallel_calls = input_reader_config.num_parallel_map_calls

        if hasattr(dataset, 'map_with_legacy_function'):
            data_map_fn = dataset.map_with_legacy_function
        else:
            data_map_fn = dataset.map


        dataset = data_map_fn(process_fn, num_parallel_calls=num_parallel_calls)

        if batch_size:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)
        return dataset

    raise ValueError()

def image_preprocess(image, height, width):
    image = image.resize((width, height))
    image_np = load_limage_into_numpy_array(image)
    thresh = threshold_local(image_np, block_size=35)
    binary = image_np > thresh

    return binary

def load_limage_into_numpy_array(image):
  limage = image.convert('L')
  (im_width, im_height) = limage.size
  return np.array(limage.getdata()).reshape(
      (im_height, im_width)).astype(np.uint8)