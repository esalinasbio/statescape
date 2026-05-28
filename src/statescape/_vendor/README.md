# Vendored third-party code

This directory contains third-party code vendored for StateScape analysis framework.

## af2rave/amino

AMINO module from [af2rave](https://github.com/tiwarylab/af2rave). AMINO is implemented as an automatic feature selection method for downstream analysis.

- Vendored: 2026-05-27 from commit 794c6a2

### Reference:

af2rave: 

- Da Teng, Vanessa J. Meraz, Akashnathan Aranganathan, Xinyu Gu, and Pratyush Tiwary, af2rave: protein ensemble generation with physics-based sampling, *Digital discovery*, 2025, 4, 2052-2061, [https://doi.org/10.1039/D5DD00201J](https://doi.org/10.1039/D5DD00201J)

AMINO:

- Pavan Ravindra, Zachary Smith and Pratyush Tiwary, Automatic mutual information noise omission (AMINO): generating order parameters for molecular systems, *Mol. Syst. Des. Eng.*, 2020, 5, 339-348, [https://doi.org/10.1039/C9ME00115H](https://doi.org/10.1039/C9ME00115H)

### License

Copyright 2024 Tiwary Lab. Licensed under the MIT License (see [LICENSE](amino/LICENSE))

### Modifications

- Removed `explanation()` and `explain()` from `wrapper.py`.

