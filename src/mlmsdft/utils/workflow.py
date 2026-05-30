# -*- coding: utf-8 -*-
"""
Helper functions for storing intermediate results in `prefect` workflows.
Long-running calculations are sometimes stopped unexpectedly, but can
be restarted from the cached intermediates.
"""
import os
import hashlib


def cache_key_function(context, parameters) -> str: # noqa: F841
    """
    Prefect workflows can cache intermediate results so that the calculation
    can be restarted. Given a set of function arguments in the dictonary
    `parameters`, the key function assigns a hash to the input combination.

    see https://docs.prefect.io/v3/concepts/caching

    NOTE: Cached results are stored in ~/.prefect/storage/

    :param context: metadata with attributes task_run_id, flow_run_id, and task
    :type context: prefect.TaskRunContext

    :param parameters: dictionary of input values to the task
    :type parameters: dict

    :return cache_key: A key that allows to distinguish if the inputs to the
        task have changed
    :rtype: str
    """
    keys = []
    for arg_name, arg in parameters.items():
        if " at 0x" in repr(arg):
            # The default string representation of a function/class instance depends on its
            # memory location, i.e. '<function f at 0x7f82cd817ce0>', which changes with
            # each run. For the hash we choose only the name, which remains constant
            # between runs.
            if hasattr(arg, "__name__"):
                name = arg.__name__
            elif hasattr(arg, "__class__"):
                name = arg.__class__.__name__
            else:
                name = str(arg)
            keys.append(f"{arg_name}={name}")
        else:
            keys.append(f"{arg_name}={repr(arg)}")
    key = "-".join(keys)
    # Hash string of function argument values into fixed length cache key.
    cache_key = hashlib.sha1(key.encode()).hexdigest()
    # Directoy where cache is stored.
    local_storage = os.environ.get("PREFECT_LOCAL_STORAGE_PATH", "~/.prefect/storage")
    print(f"NOTE: Results are chached in {local_storage} with key {cache_key}")

    return cache_key
