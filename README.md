# Transmittance Multispectral Imaging for Microcystis and Microcystin Estimation

> **Status:** Manuscript in preparation for journal submission.

## Overview

This repository accompanies the research:

**"Transmittance Multispectral Imaging for Integrated Estimation of *Microcystis* spp. Cell Concentration and Microcystin Concentration in Water"**

This work investigates a rapid, non-invasive, and cost-effective optical sensing framework for the integrated assessment of cyanobacterial abundance and toxin-related risk in water.

A custom-built transmittance multispectral imaging (MSI) system is used to acquire spectral information from *Microcystis* spp. samples. The proposed framework independently estimates:

- *Microcystis* spp. cell concentration
- ELISA-equivalent microcystin concentration

from the same multispectral acquisition.

## Methodology

The proposed framework consists of:

1. Transmittance multispectral image acquisition
2. Spectral preprocessing
3. Principal Component Analysis (PCA)
4. Wasserstein-distance-based spectral descriptor construction
5. Regression-based concentration estimation
6. Independent validation on unseen concentration levels

The MSI system acquires 12 spectral bands spanning the near-ultraviolet to near-infrared region:

`365, 405, 473, 530, 621, 660, 735, 770, 830, 850, 890, and 940 nm`

## Key Results

The framework achieved strong predictive performance for both monitoring targets.

| Target | Training R² | Validation R² |
|---|---:|---:|
| *Microcystis* spp. cell concentration | 0.9815 | 0.9809 |
| ELISA-equivalent microcystin concentration | 0.9761 | 0.9753 |

These results demonstrate the potential of transmittance MSI for rapid assessment of cyanobacterial abundance and toxin-related concentration from controlled water samples.

## Repository Contents

The implementation, experiment scripts, and result-reproduction instructions will be added to this repository.

## Code

Code for preprocessing, feature construction, regression analysis, evaluation, and figure generation will be made available here.

## Data

Information regarding the experimental dataset and its availability will be provided alongside the code release.

## Citation

Citation information will be added following publication of the associated manuscript.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Senith Jayakody**  
Department of Electrical and Electronic Engineering  
University of Peradeniya, Sri Lanka