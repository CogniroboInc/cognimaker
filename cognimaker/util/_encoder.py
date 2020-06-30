import numpy
import json
import dataclasses


class NumpyEncoder(json.JSONEncoder):
    """
    Support encoding numpy numeric and array types to JSON.
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class DataclassEncoder(json.JSONEncoder):
    """
    Support encoding dataclasses to JSON.
    """
    def default(self, obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        else:
            return super().default(obj)


class DefaultEncoder(NumpyEncoder, DataclassEncoder, json.JSONEncoder):
    pass
