This artefact contains the datasets and code for the paper "Fast Derivation of Shapley based Feature Importances through Feature Extraction Methods for Nanoinformatics".

This artefact makes use of jupyter notebooks and has the following python dependencies:
	- shap
	- sklearn
	- pandas
	- numpy
	- matplotlib

Experimental notebooks are found under '/experiments/experiments/' and each of the notebooks are appropriately named and produce the diagrams found in the paper. LibHelperFuncs.py contains helper functions used as part of a larger project. 
Note: the notebooks should be run with global_trees=1000 in order to produce the same diagrams as found in the paper.

Data can be found under '/data' and the datasets are:
	- Dataset A: "Graphene_Oxide_Bulk.csv"
	- Dataset B: "Graphene_Oxide_Nanoflake.csv"
	- Au data: "Au_nospectra.csv"
A header list for dataset A is provided and is sourced from the publicly available data source.  