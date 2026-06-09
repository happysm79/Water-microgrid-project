# Water Microgrid Pareto Sectorization

This repository contains a public, reproducible Jupyter Notebook for Water Microgrid / water distribution network sectorization analysis. The workflow evaluates candidate sectorization solutions using hydraulic, operational, and Pareto-performance indicators, and generates visual outputs such as final sectorization maps and radar charts for candidate comparison.

## Repository Structure

```text
water-microgrid-pareto-sectorization/
├── notebooks/
│   └── WMicroGrid_github_single_cell.ipynb
├── data/
│   └── README.md
├── figures/
│   └── pareto_radar_example.png
├── outputs/
│   └── .gitkeep
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Main Notebook

The main notebook is:

```text
notebooks/WMicroGrid_github_single_cell.ipynb
```

It is arranged as a single executable code cell to make the complete workflow easier to run and archive in a public GitHub repository.

## Required Input Data

Place the required EPANET input file or other network input files inside the `data/` folder before running the notebook.

Typical required input:

```text
data/Net3.inp
```

If your notebook uses a different input filename or path, update the configuration/path section near the top of the notebook before running.

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

For Conda users:

```bash
conda create -n wmicrogrid python=3.10 -y
conda activate wmicrogrid
pip install -r requirements.txt
```

## How to Run

1. Clone or download this repository.
2. Put the required `.inp` network file inside the `data/` folder.
3. Open the notebook:

```bash
jupyter notebook notebooks/WMicroGrid_github_single_cell.ipynb
```

4. Run the notebook cell.
5. Generated results will be saved to the output folders defined inside the notebook.

## Outputs

The notebook can generate outputs such as:

- Final network sectorization figure
- Boundary/closed-pipe visualization
- Meter/open-pipe visualization
- Pareto candidate comparison table
- Radar chart for Pareto candidate comparison
- Supporting result files and figures

An example radar chart is included in `figures/pareto_radar_example.png`.

## Notes for Public Use

This public version does not include private files, personal identifiers, or confidential project materials. Users should provide their own network input data and verify all paths before running.

## Citation

If you use this workflow in academic work, cite the repository and the relevant hydraulic modeling/network sectorization literature used in your study.
