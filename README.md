[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pinshuai/jupyter-for-geoscience/master?urlpath=lab/tree/notebooks/index.ipynb)

# A collection of Jupyter notebooks for geoscientists

## Motivation

Researchers often spend large amount of time writing boilerplate code for simple tasks such as visualizing and analyzing large datasets for their research. Although there are a number of existing R or Python packages for specific purpose, it is often difficult for researchers with less programming experience to choose the right package, not to mention using the package. Further, many packages lack good documentations and the example cases are often too simple compared to the real data. 

Within the geoscience community, researchers are now facing enormous amount of observation data (e.g., rainfall, streamflow and etc.) and how to visualize and analyze the data efficiently has always been a challenge. This needs to develop an easy-to-use tool for geoscientists to visualize and analyze large spatial-temporal data through interactive computational notebooks.

## Approach
We aim to provide a collection of Jupyter notebooks with each performing specific task for researchers within the geoscience community. Jupyter, an open-source, interactive, web-based notebook, has become an increasingly popular tool to conduct scientific research. It combines software code, narrative text and computational outputs in a single document. Researchers with the Jupyter notebook can easily rerun and reproduce the previous results without prior knowledge of the programming language. In addition, with the convenience of container like Docker image, researchers can launch Jupyter notebook server and run the interactive computing tools without installing any packages or dependencies. More conveniently, Jupyter notebooks can be launched instantly through cloud based free hosting service such as Binder (mybinder.org) and Google Colab (colab.research.google.com).

Our goal is to help researchers gain scientific insights from the raw data much quicker and easier. Our approach has the following advantages:

- Jupyter notebooks provides detailed documentation and friendly user interface that are easy to follow 
- Jupyter notebooks can be easily integrated into user’s existing workflow
- All computation and analysis are fully reproducible
- All notebooks are free and open-source

## Examples
This collection of notebooks are targeted at hydrologic and hydrogeologic dataset in particular. However, the workflow and methodology can be applied to geology, geophysics, atmospheric science and other field. Here are some notebooks example and the list is not exhaustive.


- Download and visualize hydrologic data from USGS 
- [wavelet.ipynb](wavelet.ipynb): Spectral analysis of time-series data using Wavelet Transform
- Interpolation of spatial data using various geostatistical methods
- Multi-variate analysis
- Groundwater flow simulation with FloPy

## Benefits

The benefits of this work may include:

1. The Jupyter notebooks provide researchers a convenient tool to visualize and analyze large datasets without coding. 
2. The Jupyter notebooks will be freely available on Github and Docker Hub, thus anyone who is interested in data visualization and data analysis may use and benefit from our work. 
3. This work would be of great interest to the geoscience community and a publication would broaden PNNL’s exposure within the community. 

## Acknowledgement
This project is partly funded by the QuickStarter program at PNNL.