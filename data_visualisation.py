import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset from csv file from panda
dataset = pd.read_csv('Customers.csv')

# shape of our dataset: 
dataset.shape 

# statistical analysis of our dataset: 
dataset.describe()

# Show types of columns
print(dataset.dtypes)

# Check total rows and columns
print(dataset.info())

# find any missing values
print(dataset.isnull().sum())

# removing id column, not needed
dataset.drop(['CustomerID'], axis=1, inplace=True)

# see updated data from the start
print(dataset.head())

# Visualisation with matplotlib
def data_visual():
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
    sns.histplot(dataset[x], bins=20, stat="density", alpha=0.5, kde=True, kde_kws={"cut": 3})
    
    plt.title('Histplot for {}'.format(x))

  # # Save figure to file .png
  # plt.savefig('dataset_visual.png')

  # Show plots
  plt.show()


if __name__ == "__main__":
    data_visual()