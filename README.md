# Scientific Python and Geographic Data

This discussion is going to highlight some of the packages and tools used in a typical scientific python environment as well as work through examples using geographic data. 

## Option 1 - Use Docker (Easiest starting from scratch)

<details>
<summary>Docker Details</summary>

1. Install [Docker](https://www.docker.com/get-started)
2. Install [Docker Compose](https://docs.docker.com/compose/install/)
3. Clone the repo, `git clone http://github.com/scottypate/spaceapps-scipy`
4. Build the Docker container, `docker-compose up --build -d`
5. To stop the container, `docker-compose down`
</details>

## Option 2 - Use Anaconda/Miniconda (Easiest if you don't use Docker.)
<details>
<summary>Anaconda/Miniconda Details</summary>

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/), or install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Install spatial index, `brew spatialindex`.
3. Clone the repo, `git clone http://github.com/scottypate/spaceapps-scipy`
4. Create a new conda environment and install dependencies, `conda create -n spaceapps python=3 --file requirements.txt`
5. Activate the new conda env, `conda activate spaceapps`
6. Start jupyter lab, `jupyter lab`
</details>

### Typical Scientific Python Data Packages

1. Scikit - Package for data processing and machine learning. 
2. Numpy - Matrix operations.
3. Pandas - Data loading and transformation.
4. Tensorflow - Tensor operations for neural networks. 

### Geographic Python Packages

1. Geopandas - Extends pandas data manipulation to geographic data
2. Shapely - Python utility to manipulate geometric data.
3. Rtree - spatial indexing for python

### Hurricane Katrina Example - [Source](https://www.datacamp.com/community/tutorials/geospatial-data-python)

Use various libraries shown here to examine data from Hurricane Katrina.

### Predicting Hurricane Path. 

Can we use past hurricane data to determine the path of a new hurricane?

*[Source](https://arxiv.org/abs/1802.02548)
*[Source](https://medium.com/@kap923/hurricane-path-prediction-using-deep-learning-2f9fbb390f18)
*[Source](https://pdfs.semanticscholar.org/cb33/81448d1e79ab28796d74218a988f203b12ee.pdf)