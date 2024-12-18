import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, davies_bouldin_score
import argparse


# Inspect the dataset
def inspect_data(data):
  # shape of our dataset: 
  data.shape 

  # statistical analysis of our dataset: 
  data.describe()

  # Show types of columns
  print(data.dtypes)

  # Check total rows and columns
  print(data.info())

  # find any missing values
  print(data.isnull().sum())

  # see updated data from the start
  print(data.head())

# Visualisation with matplotlib
def data_visual(clean_dataset):
  plt.figure(1, figsize=(12, 4))

  n = 0

  # We loop through the columns and create subplot for each
  for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1

    # Start a visual plot in a row with 3 separate visuals
    plt.subplot(1,3, n)
    # Minor adjustments
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # Use histplot with KDE line, 
    sns.histplot(clean_dataset[x], bins=20, stat="density", alpha=0.5, kde=True, kde_kws={"cut": 3})
    
    plt.title('Histplot for {}'.format(x))

  # # Save figure to file .png
  plt.savefig('dataset_visual.png')

  # Show plots
  plt.show()

# Gender Count Plot
def plot_gender(data):
  # fig, ax = plt.subplots(figsize=(15, 5))
  # sns.countplot(x='Gender', data=data, ax=ax)
  # ax.set_title('{} Distribution'.format('Gender'))
  # plt.show()
  plt.figure(figsize=(15, 5))
  sns.countplot(y='Gender', data=data)
  plt.title('Gender Distribution')
  # plt.savefig('gender_plot.png')
  plt.show()

# Age Distribution Plot
def age_group(data):
  # Make the age groups
  age_18_25 = data.Age[(data.Age >= 18) & (data.Age <= 25)]
  age_26_35 = data.Age[(data.Age >= 26) & (data.Age <= 35)]
  age_36_45 = data.Age[(data.Age >= 36) & (data.Age <= 45)]
  age_46_55 = data.Age[(data.Age >= 46) & (data.Age <= 55)]
  age_above_55 = data.Age[(data.Age >= 56)]
  # create an array for each age group/classification
  age_x = ['18-25', '26-35', '36-45','46-55','55+']
  # Create the y-axis and find the length of each age group by the count
  age_y = [len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_above_55.values)]
  # Do the plot    
  plt.figure(figsize = (15,6))
  # Use bar graphs and assign the array values to x & y axis
  sns.barplot(x = age_x, y = age_y , palette='plasma')
  plt.title('Customer Age Distribution')
  plt.xlabel('Age')
  plt.ylabel('Number of Customers')
  plt.savefig('age.png')
  plt.show()

# Annual Income Distribution Plot
def annual_income(data):
  # Creating groups for ‘Annual Income’ column and visualizing it:
  income_0_30 = data['Annual Income (k$)'][(data['Annual Income (k$)'] >= 0 ) & (data['Annual Income (k$)'] <= 30)]
  income_31_60 = data['Annual Income (k$)'][(data['Annual Income (k$)'] >= 31 ) & (data['Annual Income (k$)'] <= 60)]
  income_61_90 = data['Annual Income (k$)'][(data['Annual Income (k$)'] >= 61 ) & (data['Annual Income (k$)'] <= 90)]
  income_91_120 = data['Annual Income (k$)'][(data['Annual Income (k$)'] >= 91 ) & (data['Annual Income (k$)'] <= 120)]
  income_121_150 = data['Annual Income (k$)'][(data['Annual Income (k$)'] >= 121 ) & (data['Annual Income (k$)'] <= 150)]

  annual_x = ['$ 0-30,000','$ 31,000-60,000','$ 61,000-90,000','$ 91,000-120,000','$ 121,000-150,000']
  annual_y = [len(income_0_30.values),len(income_31_60.values),len(income_61_90.values),len(income_91_120.values),len(income_121_150.values)]

  plt.figure(figsize=(15,6))
  # legend=False added to prevent warning popup
  sns.barplot(x = annual_x, y = annual_y, palette='Spectral', legend=False)
  plt.title('Annual Income Distribution')
  plt.xlabel('Income')
  plt.ylabel('Number of Customer')
  # plt.savefig('<name_of_function>.png')
  plt.show()

def spending_score(data):
   # Creating groups of ‘Spending Score’ column and visualizing it:
  ss_1_20 = data['Spending Score (1-100)'][(data['Spending Score (1-100)'] >= 1) & (data['Spending Score (1-100)'] <= 20)]
  ss_21_40 = data['Spending Score (1-100)'][(data['Spending Score (1-100)'] >= 21) & (data['Spending Score (1-100)'] <= 40)]
  ss_41_60 = data['Spending Score (1-100)'][(data['Spending Score (1-100)'] >= 41) & (data['Spending Score (1-100)'] <= 60)]
  ss_61_80 = data['Spending Score (1-100)'][(data['Spending Score (1-100)'] >= 61) & (data['Spending Score (1-100)'] <= 80)]
  ss_81_100 = data['Spending Score (1-100)'][(data['Spending Score (1-100)'] >= 81) & (data['Spending Score (1-100)'] <= 100)]
  
  ss_x = ['1-20','21-40','41-60','61-80','81-100']
  ss_y = [len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]
  
  plt.figure(figsize=(15,6))
  sns.barplot(x = ss_x, y = ss_y, palette='rocket', legend=False)
  plt.title('Spending Score Distribution')
  plt.xlabel('Score')
  plt.ylabel('Number of Customer')
  # plt.savefig('<name_of_function>.png')
  plt.show()


# Relation plot for Annual Income vs. Spending Score
def rel_plot_income_spending_score(data):
   sns.relplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data, kind='scatter', height=7, aspect=1.7, hue=None)
   plt.grid()
   plt.title('Relation between Income vs. Spending Score')
   plt.savefig('rel_income_ss.png')
   plt.show()


# Clusters based on Age & Spending Score
def cluster_age_ss(age_score_data):

   silhouette_scores = []
   davies_bouldin_indices = []
   wcss = []
   
   # Loop through a range of K values
   for k in range(2, 11):  # Silhouette and Davies-Bouldin need > 2 clusters
       # Create and fit KMeans model
       kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
       labels = kmeans.fit_predict(age_score_data)
       
       # WCSS
       wcss.append(kmeans.inertia_)
       
       # Calculate Silhouette score
       silhouette = silhouette_score(age_score_data, labels)
       silhouette_scores.append(silhouette)
       
       # Calculate DB Index
       davies_bouldin = davies_bouldin_score(age_score_data, labels)
       davies_bouldin_indices.append(davies_bouldin)
   

   plt.figure(figsize=(12, 6))
   plt.plot(range(2, 11), wcss, marker='o', label='WCSS')
   plt.title('Elbow Method for Optimal K (WCSS)')
   plt.xlabel('Number of Clusters (K)')
   plt.ylabel('WCSS')
   plt.legend()
   plt.grid(True)
   plt.show()
   
   # Silhouette score
   plt.figure(figsize=(12, 6))
   plt.plot(range(2, 11), silhouette_scores, marker='o', label='Silhouette Score', color='green')
   plt.title('Silhouette Score for Optimal K')
   plt.xlabel('Number of Clusters (K)')
   plt.ylabel('Silhouette Score')
   plt.legend()
   plt.grid(True)
   plt.show()
   
   # DB Index
   plt.figure(figsize=(12, 6))
   plt.plot(range(2, 11), davies_bouldin_indices, marker='o', label='Davies-Bouldin Index', color='purple')
   plt.title('Davies-Bouldin Index for Optimal K')
   plt.xlabel('Number of Clusters (K)')
   plt.ylabel('Davies-Bouldin Index')
   plt.legend()
   plt.grid(True)
   plt.show()
   

   optimal_k_silhouette = np.argmax(silhouette_scores) + 2  
   print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette}")
   
   
   optimal_k_davies_bouldin = np.argmin(davies_bouldin_indices) + 2
   print(f"Optimal K based on Davies-Bouldin Index: {optimal_k_davies_bouldin}")

# Kmeans model on Age & Spending Score
def kmean_age_ss(age_score_data):
   # Based on an arbitrary guess from a visual review of the curve n_clusters = 4
   kmeans = KMeans(n_clusters = 4)
   label = kmeans.fit_predict(age_score_data)
   
   print(label)
   
   print(kmeans.cluster_centers_)
   
   plt.scatter(age_score_data[:,0],age_score_data[:,1], c=kmeans.labels_,cmap = 'rainbow')
   plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = 'black')
   plt.title('Kmeans Age vs. Spending Score')
   plt.xlabel('Age')
   plt.ylabel('Spending Score (1-100)')
   plt.savefig('kmean_age_ss.png')
   plt.show()
# Clusters based on Income & Spending Score
def cluster_income_ss(income_score_data):

    wcss = []
    silhouette_scores = []
    davies_bouldin_indices = []
    
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(income_score_data)
        wcss.append(kmeans.inertia_)
        
        if k > 1:  
            labels = kmeans.labels_
            silhouette = silhouette_score(income_score_data, labels)
            davies_bouldin = davies_bouldin_score(income_score_data, labels)
            silhouette_scores.append(silhouette)
            davies_bouldin_indices.append(davies_bouldin)
    
    # Plot WCSS (Elbow Method)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8', label='WCSS')
    plt.title('Number of Clusters vs WCSS for Income vs Spending Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.legend()
    plt.grid(True)
    plt.show()

    if silhouette_scores:
        plt.figure(figsize=(12, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o', label='Silhouette Score', color='green')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True)
        plt.show()

    if davies_bouldin_indices:
        plt.figure(figsize=(12, 6))
        plt.plot(range(2, 11), davies_bouldin_indices, marker='o', label='Davies-Bouldin Index', color='purple')
        plt.title('Davies-Bouldin Index vs Number of Clusters')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Davies-Bouldin Index')
        plt.legend()
        plt.grid(True)
        plt.show()

    if silhouette_scores:
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette}")
    if davies_bouldin_indices:
        optimal_k_davies_bouldin = np.argmin(davies_bouldin_indices) + 2
        print(f"Optimal K based on Davies-Bouldin Index: {optimal_k_davies_bouldin}")

# Cluster for all columns 
def cluster_all(all_data_columns):
 # Lists to store metrics
 wcss = []
 silhouette_scores = []
 davies_bouldin_indices = []
 
 # Loop through a range of K values (starting from 2 for silhouette and Davies-Bouldin)
 for k in range(1, 11):
     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
     kmeans.fit(all_data_columns)
     wcss.append(kmeans.inertia_)
     
     if k > 1:  # Silhouette and Davies-Bouldin require at least 2 clusters
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(all_data_columns, labels))
            davies_bouldin_indices.append(davies_bouldin_score(all_data_columns, labels))
    
 # Plot WCSS (Elbow Method)
 plt.figure(figsize=(12, 6))
 plt.grid()
 plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8')
 plt.title('Number of Clusters vs. WCSS')
 plt.xlabel('Number of Clusters (K)')
 plt.ylabel('WCSS')
 plt.show()
 
 # Plot Silhouette Scores
 plt.figure(figsize=(12, 6))
 plt.grid()
 plt.plot(range(2, 11), silhouette_scores, linewidth=2, color='blue', marker='o')
 plt.title('Number of Clusters vs. Silhouette Score')
 plt.xlabel('Number of Clusters (K)')
 plt.ylabel('Silhouette Score')
 plt.show()
 
 # Plot Davies-Bouldin Index
 plt.figure(figsize=(12, 6))
 plt.grid()
 plt.plot(range(2, 11), davies_bouldin_indices, linewidth=2, color='green', marker='s')
 plt.title('Number of Clusters vs. Davies-Bouldin Index')
 plt.xlabel('Number of Clusters (K)')
 plt.ylabel('Davies-Bouldin Index')
 plt.show()

def kmean_all(all_data_columns, data):
   # Based on an arbitrary guess from a visual review of the curve n_clusters = 7
   kmeans = KMeans(n_clusters=7)
   
   label = kmeans.fit_predict(all_data_columns)

   print(label)

   print(kmeans.cluster_centers_)

   clusters = kmeans.fit_predict(all_data_columns)
   data['label'] = clusters

   fig = plt.figure(figsize=(20, 10))
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(data.Age[data.label == 0], data['Annual Income (k$)'][data.label == 0], data['Spending Score (1-100)'][data.label == 0], c='blue', s=60)
   ax.scatter(data.Age[data.label == 1], data['Annual Income (k$)'][data.label == 1], data['Spending Score (1-100)'][data.label == 1], c = 'red', s = 60)
   ax.scatter(data.Age[data.label == 2], data['Annual Income (k$)'][data.label == 2], data['Spending Score (1-100)'][data.label == 2], c = 'green', s = 60)
   ax.scatter(data.Age[data.label == 3], data['Annual Income (k$)'][data.label == 3], data['Spending Score (1-100)'][data.label == 3], c = 'orange', s = 60)
   ax.scatter(data.Age[data.label == 4], data['Annual Income (k$)'][data.label == 4], data['Spending Score (1-100)'][data.label == 4], c = 'purple', s = 60)
   ax.scatter(data.Age[data.label == 5], data['Annual Income (k$)'][data.label == 5], data['Spending Score (1-100)'][data.label == 5], c = 'brown', s = 60)
   ax.scatter(data.Age[data.label == 6], data['Annual Income (k$)'][data.label == 6], data['Spending Score (1-100)'][data.label == 6], c = 'grey', s = 60)
   ax.view_init(30,185)

   plt.title('K-means clustering for Segmentation of Customers')
   plt.xlabel('Age')
   plt.ylabel('Annual Income')
   ax.set_zlabel('Spending Score (1-100)')
   plt.savefig('kmean_all.png')
   plt.show()

def main():
  parser = argparse.ArgumentParser(description="Customer Segmentation Script")
  parser.add_argument("--file", type=str, default='Customers.csv', required=False)

  # Specified tasks
  parser.add_argument("--inspect", action="store_true", help="Inspect the current dataset")
  parser.add_argument("--visual", action="store_true", help="Visualise the data in 3 graphs for age, income & spending score")
  parser.add_argument("--gender-plot", action="store_true", help="See the gender distribution")
  parser.add_argument("--age-plot", action="store_true", help="See the age distribution")
  parser.add_argument("--annual-income", action="store_true", help="See the annual income distribution")
  parser.add_argument("--spending-score", action="store_true", help="See the spending score distribution")
  parser.add_argument("--rel-income-ss", action="store_true", help="See relplot for annual income vs. spending score")
  parser.add_argument("--cluster-age-ss", action="store_true", help="See number of clusters/centroid groups")
  parser.add_argument("--kmean-age-ss", action="store_true", help="See number of clusters/centroid groups")
  parser.add_argument("--cluster-income-ss", action="store_true", help="See number of clusters/centroid groups")
  parser.add_argument("--kmean-income-ss", action="store_true", help="See number of clusters/centroid groups")
  parser.add_argument("--cluster-all", action="store_true", help="See number of clusters/centroid groups")
  parser.add_argument("--kmean-all", action="store_true", help="See number of clusters/centroid groups")
  args = parser.parse_args()

  # load dataset from csv file from panda
  dataset = pd.read_csv('Customers.csv')

  # removing id column, not needed
  # inplace=True
  clean_dataset = dataset.drop(['CustomerID'], axis=1)

  # Age & score column as Numpy array
  X1 = clean_dataset.loc[:, ['Age', 'Spending Score (1-100)']].values

  # Income & score column as Numpy array
  X2 = clean_dataset.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values

  # All columns
  X3 = clean_dataset.iloc[:,1:]

  # Perform tasks based on arguments
  if args.inspect:
      inspect_data(clean_dataset)
  if args.visual:
      data_visual(clean_dataset)
  if args.gender_plot:
      plot_gender(clean_dataset)
  if args.age_plot:
      age_group(clean_dataset)
  if args.annual_income:
      annual_income(clean_dataset)
  if args.rel_income_ss:
      rel_plot_income_spending_score(clean_dataset)
  if args.cluster_age_ss:
      cluster_age_ss(X1)
  if args.kmean_age_ss:
      kmean_age_ss(X1)
  if args.cluster_income_ss:
      cluster_income_ss(X2)
  if args.kmean_income_ss:
      kmean_income_ss(X2)
  if args.cluster_all:
      cluster_all(X3)
  if args.kmean_all:
      kmean_all(X3, clean_dataset)

if __name__ == "__main__":
  main()