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
    Extension("feature_computation.graph_features_manthey_alfonso", ["feature_computation/graph_features_manthey_alfonso.pyx"],),
    Extension("feature_computation.graph_features_ansotegui", ["feature_computation/graph_features_ansotegui.pyx"], extra_compile_args=["-O3"], language="c++"),
    Extension("feature_computation.more_graph_features", ["feature_computation/more_graph_features.pyx"],),
]

setup(
    name='SATfeatPy',
    version='0.1',
    description='Python library to extract features from SAT problems. Re-production of SATzilla feature extractor.',
    url='https://github.com/bprovanbessell/SATfeatPy',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    install_requires=[
        'networkx==2.6.3',
        # 'community==1.0.0',
        'powerlaw==1.5',
        'scikit-learn==1.0.2',
        'scipy==1.7.3'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'generate_features=generate_features:main',
        ],
    },
    zip_safe=False,
)
