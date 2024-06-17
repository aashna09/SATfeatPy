# SAT-features

Python library to extract features from SAT problems. Re-production of SATzilla feature extractor.

Features implemented:
1-48. of Satzilla paper [1].

Fractal dimension of CVIG and VIG, modularity of VIG, and alpha (powerlaw fit of variable occurrences) from [2].

Graph features of and Recursive Weight Heuristic from [3].

(see [features](documentation/features.md))

Current command line usage, from project root directory.
This will print a dictionary of the features. Abbreviations can be found in (see [features](documentation/features.md)). [TO DO].
```
python generate_features.py path/to/cnf_file.cnf
```

N.b. current implementation reliant on SatELite binaries, which only work on linux systems.

**Dependencies**
- Python 3.8  
- NetworkX 2.6.3  
- community 1.0.0 
- powerlaw 1.5  
- sklearn 1.0.2  
- scipy 1.7.3  

A binary for ubcsat is included in the ubcsat folder, however you may have to compile and add this yourself for full functionality.
Please clone and compile [ubcsat](https://github.com/dtompkins/ubcsat/tree/beta), and put the resulting binary in the ubcsat folder, if the current binary does not work. We found the beta branch to be stable.

## How to use this package
Download dependencies
Clone this repository within your project.
You can call the feature generation from the command line by invoking the generate features python file
```python generate_features.py```
You can specify the features to generate from within this file or via command line 
TODO: add command line specification for file, different groups of features.

The steps below illustrate a brief example of how to use this tool.
1. Clone SATfeatPy repository https://github.com/bprovanbessell/SATfeatPy.
2. Follow instructions on README of SATfeatPy.
3. Create a ```sat_instance object``` from the CNF to extract features from.
4. Extract the sets of features desired by calling the functions of the instance:
e.g. ```sat_instance.gen_basic_features().```
5. The features generated can be found in the internal dictionary ```sat_instance.features_dict.```

## Important references ##
[1] Lin Xu, Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Satzilla: portfolio-based algorithm selection for sat. 
Journal of artificial intelligence research, 32:565–606, 2008  
[2] Carlos Ansótegui, Maria Luisa Bonet, Jesús Giráldez-Cru, and Jordi Levy. Structure features 
for sat instances classification. Journal of Applied Logic, 23:27–39, 2017  
[3] Enrique Matos Alfonso and Norbert Manthey. New cnf features and formula classification. In
POS@ SAT, pages 57–71, 2014

## How to cite ##
If you use SATfeatPY, please cite the following paper:\
[1] Benjamin Provan-Bessell, Marco Dalla, Andrea Visentin, and Barry O’Sullivan, ‘SATfeatPy -- A Python-based Feature Extraction System for Satisfiability’, arXiv [cs.AI]. 2022.

For the analysis of the features, please refer:\
[2] Marco Dalla, Benjamin Provan-Bessell, Andrea Visentin, and Barry O’Sullivan, ‘SAT Feature Analysis for Machine Learning Classification Tasks’, in Proceedings of the International Symposium on Combinatorial Search, 2023, vol. 16, pp. 138–142.

