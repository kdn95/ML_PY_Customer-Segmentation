# Customer Segmentation using K-means Clustering Algorithm :bar_chart: :chart_with_downwards_trend:


## Table of Contents :card_index_dividers:
- [Overview](#overview)
- [Customer Segmentation](#what-is-customer-segmentation)
- [Kmeans Clustering](#what-is-k-means-clustering)
- [Kmeans: How it works](#how-k-means-clustering-works)
- [Requirements](#requirements)
- [Installation](#install--run)
- [How-to](#how-to-use)
## Overview :scroll:
This is my first attempt at creating machine learning project based on the K-means clustering algorithm to divide customers into their respective groups based on certain key differentiators from the data collected. 

In this project (developed in Python), the following applications will render different types of graphs and plots to provide a visual representation of the dataset provided.

The aim of this project is to demonstrate "Customer Segmentation" using the K-means clustering algorithm.

## What is Customer Segmentation? :shopping: :heavy_dollar_sign:
This is the process of dividing (typically) a company's customers into certain groups that are identified by key differentiators into their respective groups. The aim is to separate customers in order to decide how to relate to them in each group to enhance/maximise the value of the each customer to the business.

Customer segmentation also helps to improve customer service by having a better understand of the customers, as well as assist in customer loyalty and retention.

## What is K-means Clustering? :desktop_computer: :abacus:
This is a common machine learning algorithm that is used for dividing a dataset into different clusters/groups. By dividing the dataset into the a number of groups so that the data points are within a group that are most comparable to each other and different from the data points outside their group. It organises unsorted data according to patterns, parallels and variations.

## How K-means clustering works :memo:
In this algorithm, we have to specify the number of clusters/group (k) first. How do we do that? We use the "Elbow" method to find the optimal "k" (number of clusters) based on a graphical evaluation. 

This is performed by looking for the "Within-Cluster Sum of Squares" (WCSS) which is simply the sum of the square distance between all points (using Euclidean distance - length between two data points, repetitively) in a cluster against the cluster "centroid"/centre data point.

We'll initialize the centroids by randomizing the dataset and randomly selecting K data points as the centroids.

The other data points are then assigned to the different types of clusters depending which centroid it is closest to. The closest it is to a centroid, it will then join that centroid's cluster/group.

Once all data points are assigned to their clusters, the algorithm will update each centroid by finding the average position of the points assigned to it.

This will be repeated until the reassigning points and the updating centroids no longer changes.

## Features :bar_chart: :chart_with_downwards_trend:

### Dataset 
I'm currently using a sample dataset in a CSV-format file which is data from an example for a supermarket. The columns are made of Customer ID, age, gender, annual income and spending score (we assume this is provided on the basis of behaviour and purchase data). <code>( --inspect command)</code>

### Histograms
I've implemented histograms and a KDE line to show the graphical representation of the columns within the datasets using matplotlib and Seaborn to define the relationships between the columns <code>(--visual command)</code>

### Bar graph
A bar graph was also used to distinguish the count between genders, various age groups, annual income and spending score (1 - 100) <code>(--gender-plot, --age-plot, --spending-score commands)</code>

### Relational Plot
As an example, I also used a relational plot to visualise the relationship between annual income vs spending score <code>(--rel-income-ss command)</code>

### Line Plot for the Elbow Method
I've imported the KMeans class from scikit-learn library for clustering by using the WCSS against the value of 'K' which then creates a line plot to show the degree of variance of the centroids to the data points. The WCSS is expected to decrease as the K value increases. Here we will identify the "elbow" in the line for the value of K (number of clusters) <code>(--cluster-age-ss, --cluster-income-ss, --cluster-all command)</code>

### Scatter Plots for the Kmeans clustering
From a visual evaluation of the line plots, I've taken the K-value and manually used that within the Kmeans clustering algorithm for its calculation and graphical build of the scatter plots.
Here I've developed 2 different scatter plots to represent the Kmeans clustering for Age vs. Spending Score and Annual Income vs. Spending Score. <code>(--kmean-age-ss &--kmean-income-ss commands)</code>

### 3D Scatter Plot for all 3 columns
I've imported the 'Axes3D' module to create a 3D representation where I use the Spending Score as the z-label against Annual Income and Age groups. <u>As this is a 3D rendered plot, please be mindful of the GPU performance on you local device.</u> <code>(--kmean-all command)</code>


## Requirements :clipboard:
<strong> NOTE** The following instructions will be performed in Linux/Ubuntu</strong>

- <strong>Docker</strong> <code>sudo apt install docker</code>
- <strong> Docker Compose</strong> <code> sudo apt install docker-compose</code>
- numpy
- pandas
- matplotlib
- (VSCode Extension, opt) Rainbow CSV
- seaborn
- scikit-learn


## Install & Run :runner:

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

## How to use :notebook:

```
python data_visualisation.py --<command>
```
For Example:
```
python data_visualisation.py --annual-income
```

### Different types of commands for the data :technologist:
<strong> I would recommend you execute these commands one at a time</strong>
If you would like to save some of these visuals for reference, feel free to uncomment the following comment in each function:
```
plt.savefig('<name_of_function>.png')
```
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
```

## Contributors :wave:
- Khang Nguyen (me) 