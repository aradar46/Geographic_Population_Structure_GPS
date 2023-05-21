
# **Data Processing and Methodology**

## PLINK files handling 

[Link to notebook](../Scripts_notebooks/0_plink_to_countryDate_MAF.ipynb)

The PLINK files (bed, bim, and fam) were parsed using pandas_plink package and an annotation file (.anno) using pandas. The genotype and annotation dataframes [desired columns] merged based on the sample_id column into one single dataframe named `genotype`.

## Filtering
In this section, the code performs sequential filtering to remove missing data from the genotype dataframe. It removes samples with more than 70% missing data and SNPs with more than 20% missing data. It further filters out samples with more than 10% missing data and SNPs with more than 6% missing data. The data is then filtered based on the "date" column, keeping only dates before 15000 (as more than 15000 years ago has small sample sizes). The data is grouped by "country_date" (Sweden_1000, Spain_1000, Spain_2000,...). Groups with less than 2 samples are identified and removed. 


## Allele Frequency Calculation

In this section, the code calculates the Minor Allele Frequency (MAF) for each SNP within each "country_date" group.

The `get_freq function` takes a dataframe as input and calculates the allele frequencies based on the genotype data. It fills missing values with the most frequent value in the column or 2 if there is no mode. It then counts the occurrences of each situation (0:homozygosity for minor alelle, 1: heterozygosity , 2:homozygosity for major allele, and missing values) and calculates their frequencies. The MAF is calculated as the frequency of 0s * 2 + frequency of 1s * 1.

The code uses the get_freq function to calculate the MAF for each SNP within each "country_date" group. It filters the genotype dataframe based on each unique "country_date" value and processes them in parallel using multiprocessing. The results are collected and stored in a new dataframe called `df_country_date_AF`.

Finally, the resulting dataframe `df_country_date_AF` is saved as a CSV file named '`0_country_date_MAF.csv`' in the '`Data/Output`' directory.


# Clustering

[Link to notebook](../Scripts_notebooks/1_clustring.ipynb)

## data preparation

The code reads a `0_country_date_MAF.csv` from previous step and adds informative columns to it. It adds region information based on the country name. The resulting dataframe will have columns for region:west_europe, country:UK region_code: 2, date:1 means 1000 years ego, latitude, and longitude of the current political entity.



# Ensemble of Encoder-Decoders

The code implements an ensemble of autoencoders to extract a low-dimensional representation of the input data. Here is a summary of the code:

Data Preprocessing:

The code assumes that the features of interest start from the 8th column (just SNPs) of the DataFrame.  
The feature columns are extracted and converted to a NumPy array.
Optionally, I could apply preprocessing steps like scaling or normalization to the feature data but I didn't do it. as the data is already is MAF.
The data is split into training and validation sets using the train_test_split function.

**Ensemble of Autoencoders:**

The code defines an ensemble of autoencoders, where each autoencoder has its own encoder and decoder architecture.
Both the encoder and decoder are implemented as feedforward neural networks.
The sizes of the hidden layers in the encoder and decoder can be adjusted as needed.
The autoencoder models are compiled with the Adam optimizer and mean squared error (MSE) loss.

**Training the Autoencoders:**

`why training the autoencoders in an ensemble?` because each autoencoder in the ensemble is trained using a different subset of the training data. This helps capture different aspects and patterns of the data, reducing the risk of overfitting and providing a more robust representation.

Each autoencoder in the ensemble is trained using the training data.
`Early stopping` is applied to monitor the validation loss during training.
The autoencoder aims to reconstruct the input data as closely as possible.

**Extracting Encoded Data:**

After training each autoencoder, the code extracts the encoded representations of the input data using the encoder part of each autoencoder.
The encoded data from each autoencoder is stored in the encoded_data_list.

**Concatenating Encoded Representations:**

Finally, the code concatenates the encoded representations from all autoencoders along the column axis.
The result is a matrix (`my_encoded_data`) where each column represents the encoded representation from one autoencoder.

The objective of this method is to learn a compressed representation of the input data by training an ensemble of autoencoders. The ensemble approach helps capture different aspects and patterns of the data, reducing the risk of overfitting and providing a more robust representation. This compressed representation can be used for tasks like dimensionality reduction, anomaly detection, or as input features for other machine learning models.

The code makes a dataframe with 8 descriptive coulmns and merges it with the encoded data and save it as a CSV file named '`1_encoded_data_frame.csv.gz`' in the '`Data/Output`' directory.

## UMAP Clustering and distance matrix calculation

The provided code defines a function called umap_plot that plots the UMAP projection of the input data and also generates a dendrogram plot using hierarchical clustering. Here is a summary of the code:

**UMAP Projection:**

The code applies UMAP (Uniform Manifold Approximation and Projection) to reduce the dimensionality of the input data. UMAP is a dimensionality reduction technique that aims to preserve the local structure of the data.
The UMAP parameters used in the code are as follows:

- random_state=42: Sets the random seed for reproducibility.
- n_neighbors=18: Number of neighbors to consider during UMAP computation.
- min_dist=0.8: Minimum distance between points in the UMAP projection.
- n_components=2: Number of dimensions for the UMAP projection.
- metric="euclidean": Distance metric used in UMAP computation.

The input data (ensemble encoded result) is transformed using UMAP, and the resulting 2D projection is stored in the df_umap variable.

**UMAP Scatter Plot:**

A scatter plot is created to visualize the UMAP projection.
The scatter plot shows the UMAP coordinates (df_umap[:, 0] and df_umap[:, 1]).

`The colors of the points are determined by the colors_map variable, using a rainbow colormap, corresponding to region codes of each group, so that each colors related to one region and we can see how well seperated our data based on geographical location.`

Annotations (data labels) are added to the plot, close to each point, using the data_label variable.
The figure is displayed with a title, x and y labels, and a colorbar.


**Dendrogram Plot:**

The code calculates the pairwise distances between points in the input data (df) using the Canberra distance metric. Cranberra is a distance metric that is used to measure the similarity between two points.
The distance matrix is then used to generate a linkage matrix for hierarchical clustering using the "ward" method.
The dendrogram plot is created using the dendrogram function, with the labels provided by distance_matrix.index.


To use the umap_plot function, you can pass the encoded data (my_encoded_data), the data labels (df1["country"]), and the colors for the scatter plot (df1["region_code"]). This function provides a visualization of the data in the reduced-dimensional UMAP space and also shows the hierarchical clustering structure in the form of a dendrogram. This can help you identify clusters of points in the UMAP projection and also understand the relationships between the clusters.