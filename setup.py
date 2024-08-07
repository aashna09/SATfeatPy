from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("feature_computation.active_features", ["feature_computation/active_features.pyx"]),
    Extension("feature_computation.enums", ["feature_computation/enums.pyx"]),
    Extension("feature_computation.array_stats", ["feature_computation/array_stats.pyx"], include_dirs=[np.get_include()]),
    Extension("feature_computation.balance_features", ["feature_computation/balance_features.pyx"]),
    Extension("feature_computation.base_features", ["feature_computation/base_features.pyx"]),
    Extension("feature_computation.preprocessing", ["feature_computation/preprocessing.pyx"], ["c"]),
    Extension("feature_computation.parse_cnf", ["feature_computation/parse_cnf.pyx"],),
    Extension("feature_computation.local_search_probing", ["feature_computation/local_search_probing.pyx"],),
    Extension("feature_computation.graph_features", ["feature_computation/graph_features.pyx"],),
    # Extension("feature_computation.stopwatch", ["feature_computation/stopwatch.pyx"],),
    Extension("feature_computation.dpll", ["feature_computation/dpll.pyx"],),
]

setup(
    name='feature_computation',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
