from functools import wraps
import inspect

# For https://github.com/pytest-dev/pytest/blob/2.7.2/_pytest/python.py#L340
# Apparently Cython injects its own __test__ attribute that's {} by default.
# bool({}) == False, and py.test thinks the developer doesn't want to run
# tests from this module.
__test__ = True

def cython_wrap(signature):
    ''' Wrap a Cython test function in a pure Python call, so that py.test
    can inspect its argument list and run the test properly.

    Source: http://stackoverflow.com/questions/32250450/'''
    sig = inspect.signature(signature)
    code = f'lambda {",".join(sig.parameters.keys())}: func{sig}'

    def decorator(func):
        return wraps(func)(eval(code, {
            'func': func}, {}))

    return decorator(signature)
