# apex-likelihood-fitting

This repository contains the functions needed to perform the maximum likelihood estimation fitting of the halo velocity field as given in Eq. (10) in Byström et al. (2024) (https://ui.adsabs.harvard.edu/abs/2024arXiv241009149B/abstract). This fitting will return the compression velocity, dipole velocity, and the apex direction. Supplied is also a file that can be used on e.g. stellar particles from simulations to select a DESI-like footprint, as was done in the paper. A tutorial shows how to use the code and how to apply the footprint selection, using the L2M11 model from https://ui.adsabs.harvard.edu/abs/2024MNRAS.527..437V/abstract as an example. There are also functions for reproducing the Sagittarius stream selection in the paper, given in ```select_Sgr.py```.

### Author

Amanda Byström, PhD student at the Institute for Astronomy, University of Edinburgh.

### Citation

If you make use of this code in research, please cite https://ui.adsabs.harvard.edu/abs/2024arXiv241009149B/abstract.

### Contact

If there are questions or issues, please get in touch with me at Amanda.Bystrom@ed.ac.uk.

### Dependencies

You will need ```numpy, scipy, numdifftools, astropy, math, healpy, matplotlib```.
