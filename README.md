# SDLHC: Segmentation - Dictionary - Levenshten Hierachical Clustering

This repository contains the Scala / R source code that implements the time series clustering method SDLHC.


## Quick start

/!\ The R scripts will check/install the following R libraries: "stringdist","mclust","fossil","dtw","dtwclust"

The scala code must be built using maven using the file ./build.sh

*run_benchmark.r*, located in ./src/main/R/ is the script to perform benchmark runs of SDLHC on the 3 datasets.

