<p align="left" >
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# Byte-03: Time resolution of meteorological forcings

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository includes the scripts for initializing and processing MIN3P simulations designed to analyze how the time resolution of meteorological forcing affects enhanced weathering carbon dioxide removal estimates. Simulations were conducted with a branched version of MIN3P found [here](TKTK). 


## Contents

```
├── byte_util/            
    └── src/byte_util/        # helper functions
├── figures/                  # scripts to process + plot data
    └── postprocess-data/     #
├── input_data/               # data + notebooks for run setup
└── simulations/              # initialize runs and plot output
```

## Quick start

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run notebooks:
```bash
uv sync
uv notebook
```
**OR**

Install the dependencies with your package manager of choice (pip, conda, etc.)

## License
This work is licensed under the MIT license. 

## About us
CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/).