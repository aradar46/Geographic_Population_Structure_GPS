
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
# import umap.umap_ as umap
# from sklearn.manifold import TSNE






def plot_silhouette_score(X, k_min=2, k_max=20):
    '''Plot the silhouette score for a range of K values
    X: The data
    k_min: The minimum number of clusters
    k_max: The maximum number of clusters   
    '''
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
    '''Perform spectral clustering on the data
    df3: The data
    k: The number of clusters
    country_date: The country and date
    '''
    
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
    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
    # ax.set_title("Spectral Clustering ({} clusters)".format(n_clusters))
    return country_labels_S, labels





def plot_country_labels(country_labels):
    ''' Plot the country labels on a map
    country_labels: The country clusters
    '''
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
        title='',
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






# def umap_plot(df, country_date, labels):
#     ''' Plot the UMAP projection of the data
#     df: The data
#     country_date: The country_date list for the annotation'''
    
#     # df_umap = reducer.fit_transform(df)
#     df_umap = umap.UMAP(random_state=42,
#                          n_neighbors=5,
#         # min_dist=0.1,
#         # n_components=3,
#         metric='canberra').fit_transform(df)
#     # size of the figure
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Visualize the UMAP results
#     plt.scatter(df_umap[:, 0], df_umap[:, 1], alpha=0.5)
#     # color the points by their cluster assignment
#     plt.scatter(df_umap[:, 0], df_umap[:, 1], c=labels, cmap='rainbow')
#     annot2 = country_date.tolist()
#     # add a annotation very small font size and close to the point
#     for i, txt in enumerate(annot2):
#         plt.annotate(txt, (df_umap[i, 0], df_umap[i, 1]), fontsize=8, xytext=(
#             5, 2), textcoords='offset points')

#      # Add a title and labels
#     ax.set_title('UMAP projection of the dataset')
#     ax.set_xlabel('UMAP1')
#     ax.set_ylabel('UMAP2')

#     # Display the plot in Streamlit
#     st.pyplot(fig)


        
# def tsne_plot(df, country_date, labels):
#     ''' Plot the t-SNE projection of the data
#     df: The data
#     country_date: The country_date list for the annotation
#     '''
#     # Apply t-SNE to the dataset
#     tsne = TSNE(n_components=2, perplexity=(len(df)-2), learning_rate=200, n_iter=1000, random_state=42)
#     df_tsne = tsne.fit_transform(df)
#     # size of the figure
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Visualize the t-SNE results
#     plt.scatter(df_tsne[:, 0], df_tsne[:, 1], alpha=0.5)
#     # color the points by their cluster assignment
#     plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=labels, cmap='rainbow')
#     annot2 = country_date.tolist()
#     # add a annotation very small font size and close to the point
#     for i, txt in enumerate(annot2):
#         plt.annotate(txt, (df_tsne[i, 0], df_tsne[i, 1]), fontsize=8, xytext=(
#             5, 2), textcoords='offset points')

#     # Add a title and labels
#     ax.set_title('t-SNE projection of the dataset')
#     ax.set_xlabel('t-SNE1')
#     ax.set_ylabel('t-SNE2')

#     # Display the plot in Streamlit
#     st.pyplot(fig)
    
    


def clusterplotting(dic):
    df3 = pd.DataFrame.from_dict(dic, orient='index', columns=['cluster'])
    df3.reset_index(inplace=True)
    df3.rename(columns={'index': 'countrydate'}, inplace=True)
    df3['country'] = df3['countrydate'].str.split('_').str[0]
    df3['date'] = df3['countrydate'].str.split('_').str[1]
    df3['value'] = 1
# Pivot the data and create the heatmap
    clusters = df3.pivot(index='date', columns='country', values='cluster')

    fig = px.bar(df3, x='cluster', y='value', color='cluster',
             title='Countries grouped cluster', text='countrydate', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    fig = px.imshow(clusters, labels=dict(
    x="Country", y="Date", color="Cluster"), color_continuous_scale='Viridis')
    st.plotly_chart(fig)