from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("feature_computation.active_features", ["feature_computation/active_features.pyx"]),
    Extension("feature_computation.enums", ["feature_computation/enums.pyx"]),
    Extension("feature_computation.array_stats", ["feature_computation/array_stats.pyx"], include_dirs=[np.get_include()]),
    Extension("feature_computation.balance_features", ["feature_computation/balance_features.pyx"])
]

setup(
    name='feature_computation',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
