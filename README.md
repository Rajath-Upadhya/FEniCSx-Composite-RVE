# Composite RVE Transverse Tensile Analysis

This repository contains a Python implementation for simulating the transverse tensile behavior of a fiber-reinforced composite Representative Volume Element (RVE). Utilizing FEniCSx and Gmsh, the script generates a 2D mesh, solves the linear elasticity problem under a prescribed transverse strain, and evaluates both the effective mechanical properties and the stress concentration factor.

## Dependencies

Ensure the following packages are installed in your Python environment to execute the simulation:
* `dolfinx` (FEniCSx)
* `gmsh` (Python API)
* `ufl` and `basix`
* `mpi4py`
* `petsc` (via `dolfinx.fem.petsc`)
* `numpy`

## Features

* **Meshing:** Dynamically generates 2D fragments of continuous fibers embedded in a square matrix (16x16 µm) utilizing the Gmsh OpenCASCADE kernel.
* **Geometry Variations:** Includes two distinct fiber layout configurations (Case 1 and Case 2) utilizing a standardized fiber radius of 3.5 µm.
* **Constitutive Modeling:** Applies linear elasticity with distinct material properties for separate phases
* **Boundary Conditions:** Restricts rigid body motion at the corner node to allow Poisson contraction, while applying displacement-driven tensile strain (target strain = 0.1) across the X-axis boundaries.
* **Post-Processing:** Computes the Effective Transverse Modulus, isolates the peak stress within the matrix phase, and calculates the stress concentration factor (K).
* **Failure Prediction:** Estimates the overall composite transverse strength anchored against a predefined matrix strength of 80.0 MPa.

## Usage Instructions

1. Add `FEniCSx-Composite-RVE.py` to your local repository directory.
2. Open `FEniCSx-Composite-RVE.py` and modify the `case_a_or_b` variable under the Inputs section to `1` or `2` to dictate the geometric layout.
3. Run the script via the command line:
   ```bash
   python FEniCSx-Composite-RVE.py
   ```
4. Review the console output for the calculated transverse modulus, stress concentration factor, and predicted transverse strength.
5. Locate the newly generated `.xdmf` (e.g., `results_case_a.xdmf`) and associated `.h5` files in your directory to visualize the displacement and stress fields in ParaView.
