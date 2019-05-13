import copy
import functools
import tensorflow as tf

import inputs

from utils import config_utils, eval_utils
from builders import model_builder, optimizer_builder
from core import standard_fields as fields

MODEL_BUILD_UTIL_MAP = {
    'get_configs_from_pipeline_file':config_utils.get_configs_from_pipeline_file,
    # 'merge_external_params_with_configs': config_utils.merge_external_params_with_config,
    'model_fn_base': model_builder.build,
    'create_train_input_fn':inputs.create_train_input_fn,
    'create_eval_input_fn':inputs.create_eval_input_fn,
    'create_predict_input_fn':inputs.create_predict_input_fn,
}


def create_model_fn(init_model_fn, configs, hparams):
    train_config = configs['train_config']
    eval_input_config = configs['eval_input_config']
    eval_config = configs['eval_config']


    def model_fn(features, labels, mode, params=None):
        params = params or {}
        total_loss, train_op, predictions, export_outputs = None, None, None, None
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        model = init_model_fn(is_training=is_training, add_summaries=True)
        scaffold = None

        preprocessed_images = features[fields.InputDataFields.image]
        tf.logging.info('msg:{}'.format(preprocessed_images))
        predictions = model.predict(preprocessed_images)

        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            total_loss = model.loss(
                predictions,
                labels[fields.InputDataFields.label])
            global_step = tf.train.get_or_create_global_step()
            training_optimizer, optimizer_summayr_vars = optimizer_builder.build()

        if mode == tf.estimator.ModeKeys.TRAIN:

            for var in optimizer_summayr_vars:
                tf.summary.scalar(var.op.name, var)

            train_op = tf.contrib.layers.optimize_loss(
                loss=total_loss,
                global_step=global_step,
                # learning_rate=None,
                learning_rate=0.001,
                # clip_gradients=clip_gradients_value,
                # optimizer=training_optimizer,
                optimizer='Adam',
                # update_ops=model.updates(),
                # variables=trainable_variables,
                # summaries=summaries,
                name='')

        def postprocess_wrapper(predictions):
            return model.format_label(predictions)

        if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
            predictions = postprocess_wrapper(predictions)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = eval_utils.get_eval_metric_ops_for_evaluatiors(
                eval_config,
                model.format_label(predictions),
                model.format_label(labels[fields.InputDataFields.label]))

            # for var in optimizer_summayr_vars:
            #     eval_metric_ops[var.op.name] = (var, tf.no_op())

        if scaffold is None:
            # keep_checkpoint_every_n_hours = (
            #     train_config.keep_checkpoint_every_n_hours)
            saver = tf.train.Saver(
                sharded=True,
                # keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                save_relative_paths=True)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            scaffold = tf.train.Scaffold(saver=saver)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs,
            scaffold=scaffold)
    return model_fn


def create_estimator_and_inputs(run_config,
                                hparams,
                                pipeline_config_path,
                                config_override=None,
                                train_steps=None,
                                model_fn_creator=create_model_fn,
                                **kwargs):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    # merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']
    create_train_input_fn = MODEL_BUILD_UTIL_MAP['create_train_input_fn']
    create_eval_input_fn = MODEL_BUILD_UTIL_MAP['create_eval_input_fn']
    create_predict_input_fn = MODEL_BUILD_UTIL_MAP['create_predict_input_fn']
    model_fn_base = MODEL_BUILD_UTIL_MAP['model_fn_base']

    configs = get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override)
    kwargs.update({
        'train_steps':train_steps
    })
    # configs = merge_external_params_with_configs(
    #     configs, hparams, kwargs_dict=kwargs
    # )
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_on_train_config = copy.deepcopy(train_input_config)
    eval_input_config = configs['eval_input_config']
    train_data_generator_config = configs['train_data_generator_config']
    eval_data_generator_config = configs['eval_data_generator_config']

    if train_steps is None and train_config.num_steps !=0:
        train_steps = train_config.num_steps

    model_fn = functools.partial(model_fn_base, model_config=model_config)

    train_input_fn = create_train_input_fn(
        train_config=train_config,
        train_input_config=train_input_config,
        model_config=model_config
    )
    eval_input_fn = create_eval_input_fn(
        eval_config=eval_config,
        eval_input_config=eval_input_config,
        model_config=model_config
    )
    eval_on_train_input_fn = create_eval_input_fn(
        eval_config=eval_config,
        eval_input_config=eval_on_train_config,
        model_config=model_config)
    predict_input_fn = create_predict_input_fn(
        predict_input_config=eval_input_config,
        model_config=model_config,
    )

    model_fn = model_fn_creator(model_fn, configs, hparams)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    return dict(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_on_train_input_fn=eval_on_train_input_fn,
        predict_input_fn=predict_input_fn,
        train_steps=train_steps,
        train_data_generator_config=train_data_generator_config,
        eval_data_generator_config=eval_data_generator_config
    )

def create_train_and_eval_specs(train_input_fn,
                                eval_input_fn,
                                eval_on_train_input_fn,
                                predict_input_fn,
                                train_steps,
                                eval_on_train_data=False,
                                final_exporter_name='Servo',
                                eval_spec_names=None):

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=train_steps)

    # if eval_spec_names is None:
    #     eval_spec_names = [str(i) for i in range(len(eval_input_fns))]
    #
    # eval_specs = []
    # for index, (eval_spec_name, eval_input_fn) in \
    #         enumerate(zip(eval_spec_names, eval_input_fns)):
    #     if index == 0:
    #         exporter_name = final_exporter_name
    #     else:
    #         exporter_name = '{}_{}'.format(final_exporter_name, eval_spec_name)
    #     exporter = tf.estimator.FinalExporter(
    #         name=exporter_name, serving_input_receiver_fn=predict_input_fn)
    #     eval_specs.append(
    #         tf.estimator.EvalSpec(
    #             name=eval_spec_name,
    #             input_fn=eval_input_fn,
    #             steps=None,
    #             exporters=exporter))

    exporter_name = final_exporter_name
    exporter = tf.estimator.FinalExporter(
        name=exporter_name, serving_input_receiver_fn=predict_input_fn)
    eval_spec = tf.estimator.EvalSpec(
            name='eval',
            input_fn=eval_input_fn,
            steps=None,
            exporters=exporter)

    if eval_on_train_data:
        eval_spec = tf.estimator.EvalSpec(
            name='eval_on_train',
            input_fn=eval_on_train_input_fn,
            steps=None)


    return train_spec, eval_spec