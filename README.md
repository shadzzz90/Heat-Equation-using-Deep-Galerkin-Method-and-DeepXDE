# Heat Equation using Deep-Galerkin-Method and DeepXDE

This repository delves into the numerical solution of the Heat Equation using the innovative Deep Galerkin Method (DGM) and the DeepXDE library. As a benchmark, the solutions obtained using DGM are compared with those derived from the traditional Finite Difference Method (FDM).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Introduction

The Heat Equation is a fundamental partial differential equation (PDE) that describes how temperature changes over space and time. While traditional methods like the Finite Difference Method have been employed for decades to solve it, recent advances in deep learning present alternative methodologies, like the Deep Galerkin Method, offering potentially faster and more accurate solutions.

In this repository, we explore how the Deep Galerkin Method, in tandem with the DeepXDE library, can be employed to solve the Heat Equation and how these solutions stand in comparison to those obtained from the FDM.

## Installation

**Prerequisites**: Ensure you have Python 3.x installed.

1. Clone the repository:

   ```bash
   git clone https://github.com/shadzzz90/Heat-Equation-using-Deep-Galerkin-Method.git
   ```

2. Navigate into the directory:

   ```bash
   cd Heat-Equation-using-Deep-Galerkin-Method
   ```

3. (Optional) Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Each method (DGM and FDM) is encapsulated in separate Python scripts. To run any of the scripts:

```bash
python [script_name].py
```

Replace `[script_name]` with the desired script, e.g., `heat_equation_DGM.py`.

## Features

- **Deep Galerkin Method**: A contemporary approach leveraging deep learning to solve the Heat Equation.

- **Finite Difference Method**: Traditional numerical technique used as a benchmark for comparison.

- **DeepXDE Integration**: Utilizing the DeepXDE library to facilitate and streamline the solution process using DGM.


## License

This project is under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

Of course! I'll integrate a references section into the README for the specified resources. 


## References

- **Deep Galerkin Method (DGM)**:
  - Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. *Journal of Computational Physics*, 375, 1339-1364.

- **DeepXDE Library**:
  - Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2019). DeepXDE: A deep learning library for solving differential equations. *arXiv preprint arXiv:1907.04502*.
...



Happy computing!

