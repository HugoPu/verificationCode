import tensorflow as tf

def combined_static_and_dynamic_shape(tensor):
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iteration=32, back_prop=True):
    if isinstance(elems, list):
        for elem in elems:
            if not isinstance(elem, tf.Tensor):
                raise ValueError()

        elem_shapes = [elem.shape.as_list() for elem in elems]

        for elem_shape in elem_shapes:
            if (not elem_shape or not elem_shape[0] or elem_shape[0] != elem_shapes[0][0]):
                return tf.map_fn(fn, elems, dtype, parallel_iteration, back_prop)

        arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
        outputs= [fn(arg_tuple) for arg_tuple in arg_tuples]
    else:
        if not isinstance(elems, tf.Tensor):
            raise ValueError()
        elems_shape = elems.shape.as_list()
        if not elems_shape or not elems_shape[0]:
            return tf.map_fn(fn, elems, dtype, parallel_iteration, back_prop)
        outputs = [fn(arg) for arg in tf.unstack(elems)]

    if all([isinstance(output, tf.Tensor) for output in outputs]):
        return tf.stack(outputs)
    else:
        if all([isinstance(output, list) for output in outputs]):
            if all([all([isinstance(entry, tf.Tensor) for entry in output_list]) for output_list in outputs]):
                return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
    raise ValueError()