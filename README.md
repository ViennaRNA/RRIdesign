# RRIdesign: Sequence design for RNA-RNA interactions

Within the RRIdesign jupyter notebook we demonstrates the use of the Infared framework to design interacting sequences.
Specifically, we consider the regulation of the rpoS mRNA by the sRNA DsrA and design artificial 5'UTRs that place a downstream protein coding gene under control of DsrA.
The notebook illustartes the set up of design constraints for sampling sequences in Infrared, computing quality measures, constructing a suitable cost function, as well as the optimization procedure. 
Not only thermodynamic, but also kinetic folding features can be relevant. Therefor, we explains how to include kinetic folding features from RRIkinDP directly in the cost function in addition to thermodynamic features like binding energy. 

## Set up
We recommend running the notebook in a conda environment (see: https://docs.anaconda.com/free/miniconda/miniconda-install). After installing and activating Conda, the required dependencies can be easily in-
stalled using the conda create:
```
$ conda create --name rridesign \
  conda-forge::’infrared>=1.2’ \
  bioconda::’viennarna>=2.6.2’ \
  bioconda::intarna \
  bioconda::’rrikindp>=0.0.2’
$ conda activate rridesign
```


The jupyter notbook with the design approach and example files can be downloaded
from https://github.com/ViennaRNA/RRIdesign, e.g., with git clone:
```
$ git clone git@github.com:ViennaRNA/RRIdesign.git
```

To run the Jupyter notebook, ensure a notebook interface is installed, e.g., Jupyter-Lab via Conda:
```
$ conda install conda-forge::jupyterlab
```

Navigate to the downloaded git repository and start JupyterLab:
```
$ cd RRIdesign
$ jupyter lab
```

This will open JupyterLab in your web browser, where you can access the notebook
RRIdesign.ipynb containing all the scripts described in the Methods section.
