import tensorflow as tf

from metrics import evaluation

EVAL_DEFAULT_METRIC = 'accuracy'

EVAL_METRICS_CLASS_DICT = {
    'accuracy': evaluation.CodeEvaluator
}

def get_eval_metric_ops_for_evaluatiors(
        eval_config,
        predictions,
        labels):

    # eval_metric_ops = {}
    # evaluator_options = evaluator_options_from_eval_config(eval_config)
    # evaluators_list = get_evaluators(eval_config, categories, evaluator_options)
    # for evaluator in evaluators_list:
    #     eval_metric_ops.update(evaluator.get_estimator_eval_metric_ops(
    #         eval_dict))

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels, predictions)}
    return eval_metric_ops

# def evaluator_options_from_eval_config(eval_config):
#     eval_metric_fn_keys = eval_config.metrics_set
#     evaluator_options = {}
#     return evaluator_options
#
# def get_evaluators(eval_config, categories, evaluator_options=None):
#     evaluator_options = evaluator_options or {}
#     eval_metric_fn_keys = eval_config.metrics_set
#     if not eval_metric_fn_keys:
#         eval_metric_fn_keys = [EVAL_DEFAULT_METRIC]
#     evaluators_list = []
#     for eval_metric_fn_key in eval_metric_fn_keys:
#         if eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT:
#             raise ValueError()
#         kwargs_dict = (evaluator_options[eval_metric_fn_key] if eval_metric_fn_key
#                                                                 in evaluator_options else {})
#         evaluators_list.append(
#             EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories, **kwargs_dict))
#
#     return evaluators_list
