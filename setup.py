from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("feature_computation.active_features", ["feature_computation/active_features.pyx"]),
    Extension("feature_computation.enums", ["feature_computation/enums.pyx"])
]

setup(
    name='feature_computation',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
