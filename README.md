# DMV

Experimental codes for Dependency Model with Valence (Klein and Manning, 2004; Berg-KirkPatrick et al., 2010).

Currently supprted models/parsers:
    - [DMV (Klein and Manning, 2004)](https://dl.acm.org/citation.cfm?id=1219016)
    - [Log-Linear DMV (Berg-KirkPatrick et al., 2010)](https://aclweb.org/anthology/N10-1083)

## Requirements ##

- numpy
- chainer
- pyprind
- [The LTH Constituent-to-Dependency Conversion Tool for Penn-style Treebanks](http://nlp.cs.lth.se/software/treebank_converter)
- https://github.com/norikinishida/utils.git
- https://github.com/norikinishida/treetk.git
- https://github.com/norikinishida/textpreprocessor.git

## Setting ##

Edit following files:

- ```run_preprocessing.sh``` ("PATH\_PTBWSJ", "PENNCONVERTER" and  "PATH\_DEP")
- ```config/path.ini```
- ```config/experiment_26.ini```

## Preprocessing ##

```
./run_preprocessing.sh
```

## Baselines ##

```
./run_baselines.sh
```

## Training, Evaluation, Output dumping ##

```
./run_methods.sh
```

