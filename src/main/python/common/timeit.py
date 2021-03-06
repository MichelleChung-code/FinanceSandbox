import time
import functools
# timeit decorator modified from: https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d

def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        print('Time elapsed for %r function:  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed
