# Customer Segmentation using K-means Clustering Algorithm

## Table of Contents
- [Overview](#overview)
- [Customer Segmentation](#what-is-customer-segmentation)
- [Kmeans Clustering](#what-is-k-means-clustering)
- [Kmeans: How it works](#how-k-means-clustering-works)
- [Requirements](#requirements)
- [Installation](#install--run)
- [How-to](#how-to-use)
## Overview
This is my first attempt at creating machine learning project based on the K-means clustering algorithm to divide customers into their respective groups based on certain key differentiators from the data collected. 

In this project (developed in Python), the following applications will render different types of graphs and plots to provide a visual representation of the dataset provided.

The aim of this project is to demonstrate "Customer Segmentation" using the K-means clustering algorithm.

## What is Customer Segmentation?
This is the process of dividing (typically) a company's customers into certain groups that are identified by key differentiators into their respective groups. The aim is to separate customers in order to decide how to relate to them in each group to enhance/maximise the value of the each customer to the business.

Customer segmentation also helps to improve customer service by having a better understand of the customers, as well as assist in customer loyalty and retention.

## What is K-means Clustering?
This is a common machine learning algorithm that is used for dividing a dataset into different clusters/groups. By dividing the dataset into the a number of groups so that the data points are within a group that are most comparable to each other and different from the data points outside their group. It organises unsorted data according to patterns, parallels and variations.

## How K-means clustering works
In this algorithm, we have to specify the number of clusters/group (k) first. How do we do that? We use the "Elbow" method to find the optimal "k" (number of clusters) based on a graphical evaluation. 

This is performed by looking for the "Within-Cluster Sum of Squares" (WCSS) which is simply the sum of the square distance between all points (using Euclidean distance - length between two data points, repetitively) in a cluster against the cluster "centroid"/centre data point.

We'll initialize the centroids by randomizing the dataset and randomly selecting K data points as the centroids.

The other data points are then assigned to the different types of clusters depending which centroid it is closest to. The closest it is to a centroid, it will then join that centroid's cluster/group.

Once all data points are assigned to their clusters, the algorithm will update each centroid by finding the average position of the points assigned to it.

This will be repeated until the reassigning points and the updating centroids no longer changes.

## Requirements
<strong> NOTE** The following instructions will be performed in Linux/Ubuntu</strong>

- <strong>Docker</strong> <code>sudo apt install docker</code>
- <strong> Docker Compose</strong> <code> sudo apt install docker-compose</code>
- numpy
- pandas
- matplotlib
- (VSCode Extension, opt) Rainbow CSV
- seaborn
- scikit-learn


## Install & Run

### Steps
1. Clone the repository:
```
git clone https://github.com/<username>/ML_PY_Customer-Segmentation.git
```
2. cd your-repo
3. Build & start the container on the terminal line
``` 
docker-compose up --build
```
4. Check pip list on terminal line & make sure the packages in the requirements are here
```
pip list
```
5a. (Optional) If not installed, install manually from requirements.txt
```
pip install -r requirements.txt
```
5b. (Optional) or just on the terminal line
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to use

```
python data_visualisation.py --<command>
```
For Example:
```
python data_visualisation.py --annual-income
```

### Different types of commands for the data
```
- inspect
- visual
- gender-plot
- age-plot
- annual-income
- spending-score
- rel-income-ss
- cluster-age-ss
- kmean-age-ss
- cluster-income-ss
- kmean-income-ss
- cluster-all
- kmean-all