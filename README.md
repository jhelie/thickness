Please cite: [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10539.png)](http://dx.doi.org/10.5281/zenodo.10539)

thickness
=========

Python script to calculate lipid bilayer thickness from MD simulations

requirements
------------
The following Python module(s):
-MDAnalysis

usage
-----
python thickness --help

features
--------
-actual thickness based on euclidian distance (not a difference along the z axis), so deforming system can be handled
-evolution of bilayer thickness
-statistics and graphs by lipid specie
-output files for visualisation in VMD
