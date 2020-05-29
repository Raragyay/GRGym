import _pytest
import importlib


class Module(_pytest.python.Module):
    # Source: http://stackoverflow.com/questions/32250450/
    def _importtestmodule(self):
        # Copy-paste from py.test, edited to avoid throwing ImportMismatchError.
        # Defensive programming in py.test tries to ensure the module's __file__
        # matches the location of the source code. Cython's __file__ is
        # different.
        # https://github.com/pytest-dev/pytest/blob/2.7.2/_pytest/python.py#L485
        path = self.fspath
        pypkgpath = path.pypkgpath()
        modname = '.'.join(
            [pypkgpath.basename] +
            path.new(ext='').relto(pypkgpath).split(path.sep))
        print(f"``MODNAME``: {modname}")
        mod = importlib.import_module(modname)
        self.config.pluginmanager.consider_module(mod)
        return mod

    def collect(self):
        # Defeat defensive programming.
        # https://github.com/pytest-dev/pytest/blob/2.7.2/_pytest/python.py#L286
        assert self.name.endswith('.pyx')
        self.name = self.name[:-1]
        return super(Module, self).collect()


def pytest_collect_file(parent, path):
    # py.test by default limits all test discovery to .py files.
    # I should probably have introduced a new setting for .pyx paths to match,
    # for simplicity I am hard-coding a single path.
    if path.fnmatch('cytest_*.pyx'):
        return Module.from_parent(parent, fspath=path)
