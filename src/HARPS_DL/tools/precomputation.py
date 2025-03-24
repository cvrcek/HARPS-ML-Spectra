import pickle
import shelve
import hashlib



def persist_to_file(file_name):
    """ Decorator that caches the return value of a function call and writes it to a file."""

    def decorator(original_func):

        try:
            cache = pickle.load(open(file_name, 'rb'))
        except (IOError, ValueError):
            cache = {}

        def new_func(*args, **kwargs):
            params = str(args) + str(kwargs)
            if params not in cache:
                cache[params] = original_func(*args, **kwargs)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[params]

        return new_func

    return decorator

def cache_output(file_name, key):
    """ Decorator that caches the return value of a function call and writes it to a file."""

    def decorator(original_func):
        try:
            cache = pickle.load(open(file_name, 'rb'))
        except (IOError, ValueError):
            cache = {}

        def new_func(*args, **kwargs):
            if key not in cache:
                cache[key] = original_func(*args, **kwargs)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[key]

        return new_func

    return decorator