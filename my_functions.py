
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


def plot_silhouette_score(X, k_min=2, k_max=20):
    # Create a range of K values
    k_range = range(k_min, k_max+1)
    # Create an empty list to store the silhouette scores
    silhouette_scores = []
    # Loop through the range of K values and calculate the silhouette score for each value
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
    # Plot the silhouette scores for each K value
    plt.plot(k_range, silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means clustering')
    plt.xticks(np.arange(k_min, k_max+1, step=2))
    st.pyplot(plt)  # Show the plot in the app using st.pyplot()
    top_3 = sorted(range(len(silhouette_scores)),
                   key=lambda i: silhouette_scores[i], reverse=True)[:3]
    top_3 = [(k_min+x) for x in top_3]
    return top_3


def spectral_clustering(df3, k, country_date):
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df3)
    # Perform spectral clustering
    n_clusters = k  # Number of clusters
    clustering = SpectralClustering(
        n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors", random_state=42).fit(X)    
    country_labels_S = dict(zip(country_date, clustering.labels_))
    labels = clustering.labels_
    # Visualize the clustering result
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
    ax.set_title("Spectral Clustering ({} clusters)".format(n_clusters))
    return fig, country_labels_S, labels


def plot_country_labels(country_labels):
    # Define the data for the Choropleth map
    data = go.Choropleth(
        locations=list(country_labels.keys()),  # The countries
        z=list(country_labels.values()),  # The cluster labels
        locationmode='country names',
        colorscale='Viridis',  # The color scale to use
        colorbar=dict(title='Cluster'),
    )
    # Define the layout for the map
    layout = go.Layout(
        title='Cluster Labels by Country',
        geo=dict(showframe=False, showcoastlines=True,
                 projection_type='equirectangular'),
        annotations=[
            dict(
                x=0.55,
                y=0.1,
                xref='paper',
                yref='paper',
                text='Annotations Here',
                showarrow=False,
            )
        ]
    )

    # Create the figure and add the Choropleth map and layout to it
    fig = go.Figure(data=data, layout=layout)
    # Display the figure using Streamlit
    st.plotly_chart(fig)


def umap_plot(df, country_date):
    # Apply UMAP to the dataset
    reducer = umap.UMAP()
    # df_umap = reducer.fit_transform(df)
    df_umap = umap.UMAP(min_dist=0.1, random_state=21).fit_transform(df)
    # size of the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Visualize the UMAP results
    plt.scatter(df_umap[:, 0], df_umap[:, 1], alpha=0.5)
    # color the points by their cluster assignment
    # plt.scatter(df_umap[:, 0], df_umap[:, 1], c=labels, cmap='rainbow')
    annot2 = country_date.tolist()
    # add a annotation very small font size and close to the point
    for i, txt in enumerate(annot2):
        plt.annotate(txt, (df_umap[i, 0], df_umap[i, 1]), fontsize=8, xytext=(
            5, 2), textcoords='offset points')

     # Add a title and labels
    ax.set_title('UMAP projection of the dataset')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    # Display the plot in Streamlit
    st.pyplot(fig)

#%%
import graphviz
import pandas





# plot_country_cluster(country_labels)
# %%
