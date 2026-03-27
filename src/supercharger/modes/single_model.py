from supercharger.backends import *
import inspect
import logging

logger = logging.getLogger("lib.smodel")

def single_model(model: str, json_file: str, messages: list = []):
    backend_class = globals().get(model)

    if backend_class is None:
        raise ValueError(f"Backend class '{model}' not found. Available: {', '.join([k for k in globals() if not k.startswith('__')])}")

    # If a module was imported instead of the class, extract the class from it
    if not inspect.isclass(backend_class):
        backend_class = getattr(backend_class, model, None)
        if backend_class is None or not inspect.isclass(backend_class):
            raise ValueError(f"Could not find class '{model}' in the resolved module.")

    backend = (
        backend_class(json_file=json_file)
        if not messages
        else backend_class(json_file=json_file, messages=messages)
    )
    logger.info(f"using \"{backend_class.__name__}\" back end.")
    backend()
