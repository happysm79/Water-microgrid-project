# WDS Storage Tank Siting

This repository contains Python tools for hydraulically informed graph-theoretic ranking of candidate storage tank locations in water distribution systems (WDSs).

The code supports full-network and sector-level analysis of water distribution networks using EPANET/WNTR hydraulic simulation, graph theory metrics, entropy-based weighting, spectral clustering, optional DMA analysis, and IGT-based source configuration evaluation.

---

## Overview

The main objective of this repository is to identify promising candidate nodes for storage tank placement in water distribution systems using a hydraulically informed graph-theoretic framework.

The workflow includes:

1. Reading an EPANET `.inp` file
2. Extracting node and link attributes from the water distribution network
3. Running a steady-state hydraulic simulation using WNTR
4. Computing hydraulic edge weights using Hazen-Williams resistance
5. Building topological and hydraulically weighted graph representations
6. Computing graph centrality metrics
7. Applying min-max normalization and entropy-based weighting
8. Ranking candidate storage tank locations
9. Performing sector-based analysis using spectral clustering
10. Optionally evaluating predefined DMA configurations
11. Evaluating source configurations using an IGT-based metric
12. Exporting summary tables, ranking results, figures, and output files

---

## Main Features

- EPANET/WNTR-based water network loading
- Steady-state hydraulic simulation
- Hydraulic edge-weight calculation
- Topological and hydraulically weighted graph construction
- Degree centrality, closeness centrality, and betweenness centrality analysis
- Min-max normalization of metrics
- Entropy-based metric weighting
- Full-network node ranking
- Sector-level node ranking
- Spectral clustering for network sectorization
- Optional predefined DMA analysis
- IGT-based evaluation of existing and proposed source configurations
- Automated export of CSV tables and network figures
- ZIP file generation for downloading all outputs

---

## Repository Structure

```text
wds-storage-tank-location/
│
├── README.md
├── requirements.txt
├── .gitignore
├── HI_GT_storage_tank_siting.py
│
└── data/
    └── README.md
```

---

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
wntr
networkx
numpy
pandas
matplotlib
scikit-learn
openpyxl
```

---

## How to Run

This version is designed to run in Google Colab.

### Step 1: Open the Code

Open the Python file:

```text
HI_GT_storage_tank_siting.py
```

You may also copy the code into a Google Colab notebook.

### Step 2: Install Required Packages

If running in Google Colab, install the required packages by running:

```python
!pip -q install wntr openpyxl scikit-learn
```

### Step 3: Run the Script

Run the full script.

The code will ask you to upload an EPANET `.inp` file.

### Step 4: Upload Input File

Upload your EPANET input file when prompted.

Example input file format:

```text
network.inp
```

### Step 5: Download Outputs

After the analysis is completed, the code creates an output folder and downloads a ZIP file containing all generated tables and figures.

---

## Input Data

The main input is an EPANET `.inp` file representing a water distribution network.

The input file may include:

- Junctions
- Tanks
- Reservoirs
- Pipes
- Pumps
- Valves
- Node coordinates
- Elevations
- Base demands
- Pipe lengths
- Pipe diameters
- Pipe roughness values

---

## Output Files

The code generates several output tables and figures.

### Main Tables

```text
network_component_summary.csv
node_attributes_with_dma_labels.csv
hydraulic_edge_weights.csv
full_network_ranking.csv
full_network_entropy_weights.csv
full_network_selected_candidates.csv
full_network_igt_summary.csv
combined_sector_rankings.csv
combined_sector_entropy_weights.csv
combined_sector_selected_candidates.csv
combined_sector_igt_summary.csv
```

### Main Figures

```text
edge_weight_map.png
full_network_centrality_score_map.png
cluster_map.png
cluster_score_map.png
dma_score_map.png
```

---

## Methodological Summary

The framework represents the water distribution system as a graph, where nodes represent junctions, tanks, and reservoirs, and links represent pipes, pumps, and valves.

For pipes, hydraulic edge weights are calculated using a flow-independent Hazen-Williams resistance coefficient:

```text
r = 10.67 L / (C^1.852 D^4.871)
```

where:

- `L` is pipe length
- `C` is the Hazen-Williams roughness coefficient
- `D` is pipe diameter

The graph-based ranking uses the following metrics:

- Degree centrality
- Weighted closeness centrality
- Weighted betweenness centrality

These metrics are normalized and combined using entropy-based weighting to calculate a composite score for each candidate node.

Existing tanks and reservoirs are excluded from the candidate node ranking.

---

## Full-Network Analysis

The full-network analysis ranks candidate nodes across the entire water distribution system.

The output includes:

- Normalized centrality metrics
- Entropy weights
- Composite score
- Final node ranking
- Selected top candidate nodes
- IGT-based evaluation of proposed source configurations

---

## Sector-Level Analysis

The code also supports sector-level ranking using spectral clustering.

Sector-level analysis includes:

- Automatic selection of cluster number
- User-defined cluster number
- Tank-based cluster number
- Optional predefined DMA-based analysis

For each sector, candidate nodes are ranked independently using local graph metrics and entropy-based weighting.

---

## IGT-Based Evaluation

The code evaluates existing and proposed source configurations using an IGT-based metric.

The evaluation compares:

- Existing tanks and reservoirs
- Proposed candidate storage tank locations plus reservoirs

The IGT results include:

- Mean IGT
- Trimmed mean IGT
- Minimum IGT
- Normalized variance of IGT
- Reachability ratio
- Number of evaluated demand nodes

These results can support decision-making for identifying more effective source configurations.

---

## Notes

This repository is currently under development for research purposes.

Before making this repository public, make sure that:

- The code is cleaned and reviewed
- Input data are allowed to be shared
- Unpublished results are removed or approved for release
- The advisor or research team approves public sharing
- A proper license is selected if public use is intended

---

## Author

Amirmahdi Ghanaatikashani  
PhD Student  
Civil, Environmental, and Infrastructure Engineering  
Southern Illinois University Carbondale

---

## Citation

If you use this code or adapt the methodology, please cite the related research work once it becomes available.

Suggested citation format:

```text
Ghanaatikashani, A., Hasnat, A., and Shin, S. Hydraulically informed graph-theoretic framework for storage tank siting in water distribution systems.
```

---

## License

No license has been added yet.

This repository is currently intended for private research development.
