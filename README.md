
# Quantum Computing for Quantum Chemistry
<!-- (Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.) -->
[![Build Status](https://github.com/qc2nl/qc2/actions/workflows/build.yml/badge.svg)](https://github.com/qc2nl/qc2/actions)
[![Documentation Status](https://readthedocs.org/projects/qc2/badge/?version=latest)](https://qc2.readthedocs.io/en/latest/?badge=latest)
[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:qc2nl/qc2)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![RSD](https://img.shields.io/badge/rsd-qc2-00a3e3.svg)](https://www.research-software.nl/software/qc2)
[![workflow pypi badge](https://img.shields.io/pypi/v/qc2.svg?colorB=blue)](https://pypi.python.org/project/qc2/)

<!-- | fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:qc2nl/qc2) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/qc2nl/qc2)](git@github.com:qc2nl/qc2) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-qc2-00a3e3.svg)](https://www.research-software.nl/software/qc2) [![workflow pypi badge](https://img.shields.io/pypi/v/qc2.svg?colorB=blue)](https://pypi.python.org/project/qc2/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=qc2nl_qc2&metric=alert_status)](https://sonarcloud.io/dashboard?id=qc2nl_qc2) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=qc2nl_qc2&metric=coverage)](https://sonarcloud.io/dashboard?id=qc2nl_qc2) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/qc2/badge/?version=latest)](https://qc2.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](git@github.com:qc2nl/qc2/actions/workflows/build.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](git@github.com:qc2nl/qc2/actions/workflows/cffconvert.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](git@github.com:qc2nl/qc2/actions/workflows/sonarcloud.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](git@github.com:qc2nl/qc2/actions/workflows/markdown-link-check.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/markdown-link-check.yml) | -->

## About qc2

qc2 is a modular software designed to seamlessly integrate traditional computational chemistry codes
and quantum computing frameworks.
It is specifically crafted for hybrid quantum-classical workflows
such as the variational quantum eigensolver (VQE) algorithm.
The software relies on custom [ASE calculators](https://wiki.fysik.dtu.dk/ase/) as well as formatted data files
(*e.g.*, [QCSchema](https://molssi.org/software/qcschema-2/) or [FCIDUMP](https://www.sciencedirect.com/science/article/abs/pii/0010465589900337?via%3Dihub)) to efficiently offload 1- and 2-electron
integrals needed by various Python quantum computing libraries.

The qc2 software is a direct outcome of the [QCforQC project](https://research-software-directory.org/projects/qcforqc),
a collaboration between [Netherlands eScience Center](https://www.esciencecenter.nl/),
[Vrije Universiteit Amsterdam (VU)](https://research.vu.nl/en/persons/luuk-visscher) and [SURF](https://www.surf.nl/en/themes/quantum).

To access qc2's capabilities and current status, please refer to its documentation at https://qc2.readthedocs.io.

## Installation

To install qc2 from GitHub repository, do:

```console
git clone git@github.com:qc2nl/qc2.git
cd qc2
python3 -m pip install -e .
```

In this current version, qc2 can perform hybrid quantum-classical calculations using both [Qiskit Nature](https://qiskit.org/ecosystem/nature/) and [PennyLane](https://pennylane.ai/). However, the latter is an optional dependency. To install `Pennylane` and perform automatic testing with it, follow these steps:
```console
git clone git@github.com:qc2nl/qc2.git
cd qc2
python3 -m pip install -e .[pennylane] # (use ".[pennylane]" if you have zsh shell)
```

For detailed installation instructions, please refer to the [Installation Instructions](https://qc2.readthedocs.io/en/latest/get-started/installation.html) section in our documentation.

## Documentation

[https://qc2.readthedocs.io](https://qc2.readthedocs.io)

## Contributing

If you want to contribute to the development of qc2,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
