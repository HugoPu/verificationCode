import tensorflow as tf

from model import model_lib, model_hparams
from utils import data_utils

flags = tf.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string(
    'model_dir', '/sdb/hugo/PythonWorkspace/verificationCode/output', 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', '/sdb/hugo/PythonWorkspace/verificationCode/hparams/pipeline.config', 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
# flags.DEFINE_boolean('eval_training_data', False,
#                      'If training data should be evaluated for this job. Note '
#                      'that one call only use this in eval-only mode, and '
#                      '`checkpoint_dir` must be supplied.')
# flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
#                      'every n eval input examples, where n is provided.')
# flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
#                      'one of every n train input examples for evaluation, '
#                      'where n is provided. This is only used if '
#                      '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')

flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
                       'one round of eval vs running continuously (default)')

flags.DEFINE_boolean(
    'generate_tfrecords', False,''
)

FLAGS = flags.FLAGS

def main(argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps
    )

    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fn = train_and_eval_dict['eval_input_fn']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    train_data_generator_config = train_and_eval_dict['train_data_generator_config']
    eval_data_generator_config = train_and_eval_dict['eval_data_generator_config']

    if FLAGS.generate_tfrecords:
        data_utils.generate_tfrecords(train_data_generator_config)
        data_utils.generate_tfrecords(eval_data_generator_config)

    if FLAGS.checkpoint_dir:
        name = 'training_data'
        input_fn = eval_on_train_input_fn
    else:
        name = 'validation_data'
        input_fn = eval_input_fn
    if FLAGS.run_once:
        estimator.evaluate(
            input_fn,
            steps=None,
            checkpoint_path=tf.train.latest_checkpoint(
                FLAGS.checkpoint_dir))
    else:
        train_spec, eval_spec = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fn,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ =='__main__':
    tf.app.run()