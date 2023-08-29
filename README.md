# Unofficial Implementation of the Deep Galerkin Method (DGM) for PDEs

## Overview

Welcome to the unofficial code repository for the Deep Galerkin Method (DGM), a transformative deep learning algorithm developed to solve high-dimensional Partial Differential Equations (PDEs). Here, we attempt to replicate and delve into the paper's claims of leveraging deep neural networks to find approximate solutions for intricate PDEs, circumventing the pitfalls of conventional mesh-centric methodologies.

## Key Aspects

- **Mesh-Independence**: Designed to thrive in high-dimensional realms where meshes falter.
- **Adoption of Random Sampling**: Forgoing grids, DGM leans on random samples across space and time.
- **Extensive Validations**: Probed against diverse PDEs from high-dimensional free boundary ones to the Hamilton-Jacobi-Bellman PDE and Burgers’ equation.
- **Broad-Based Solutions**: Potent enough to infer a spectrum of boundary and physical conditions, with particular efficacy observed for Burgers’ equation.
- **Theoretical Insight**: Features a theorem delineating neural networks' approximation prowess for specific quasilinear parabolic PDEs.

## Setting Sail

### Preliminaries

1. Ensure [Python 3.8](https://www.python.org/downloads/) or its newer siblings are available.
2. Onboard the requisite packages:
    ```bash
    pip install -r requirements.txt
    ```

### Instructions

1. **DGM Model Training**:
   ```bash
   python train_dgm.py --epochs 5000 --batch_size 512
   ```

2. **Evaluating and Plotting**:
   ```bash
   python evaluate_dgm.py --model_path path_to_saved_model
   ```

## Interactive Jupyter Notebooks

Immerse into the DGM through our Jupyter Notebooks:

1. **[Introduction to DGM](link_to_intro_notebook.ipynb)**
2. **[Navigating High-dimensional Free Boundary PDEs](link_to_free_boundary_notebook.ipynb)**
3. **[Burgers’ Equation Decoded with DGM](link_to_burgers_notebook.ipynb)**

## Contributing 

Any suggestions, findings, or insights? Contributions are heartily encouraged! Kindly peep into [CONTRIBUTING.md](CONTRIBUTING.md) for the modus operandi.

## Acknowledgment

While this is an unofficial rendition, all credit for the original research and method goes to the paper's authors. Do cite the original work:

```bibtex
@article{authorsYearDGM,
  title={DGM: A deep learning algorithm for solving partial differential equations},
  author={Original Authors},
  journal={Original Journal},
  year={Original Year}
}
```

## Licensing

Freely available under the MIT License. Check out [LICENSE.md](LICENSE.md) for specifics.

## Get in Touch

For queries or suggestions, feel free to open an issue or ping at [email@example.com](mailto:email@example.com).

---

Embracing the meshfree revolution with DGM. Dive in and explore!# Heat-Conduction-using-Deep-Galerkin-Method
