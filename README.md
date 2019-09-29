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
2. Clone the repo, `git clone http://github.com/scottypate/spaceapps-scipy`
3. Create a new conda environment and install dependencies, `conda create -n spaceapps python=3 --file requirements.txt`
4. Activate the new conda env, `conda activate spaceapps`
5. Start jupyter lab, `jupyter lab`
</details>

### Typical Scientific Python Data Packages

1. Scikit - Package for data processing and machine learning. 
2. Numpy - Matrix operations.
3. Pandas - Data loading and transformation.
4. Tensorflow - Tensor operations for neural networks. 

### Geographic Python Packages

1. Geopandas - Extends pandas data manipulation to geographic data
2. Shapely - Python utility to manipulate geometric data.

### Hurricane Katrina Example - [Source](https://www.datacamp.com/community/tutorials/geospatial-data-python)

Use various libraries shown here to examine data from Hurricane Katrina.

### Predicting Hurricane Path. 

Can we use past hurricane data to determine the path of a new hurricane?