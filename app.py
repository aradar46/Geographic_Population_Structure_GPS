
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


st.title("MAF Viewer")

# Allow user to upload a CSV file or load a default file
uploaded_file = st.file_uploader("Choose a csv.gz file", type="gz")
if uploaded_file is None:
    # st.info("No file uploaded. Loading default file...")
    df1 = pd.read_csv('Data/1_raw2freq/test_1.csv.gz', sep="\t", compression='gzip')
    df1= pd.DataFrame(df1)
  
    
else:
    df1 = pd.read_csv(uploaded_file, sep="\t", compression='gzip', header=None)
    df1= pd.DataFrame(df1)
    
# Display the dataframe in the app
st.write(df1.head(10))


df=df1
# Sidebar for user input
st.sidebar.title("Country and Date Selector")
select_all_countries = st.sidebar.checkbox("Select all countries", value=False)
if select_all_countries:
    selected_countries = df["country"].unique()
else:
    selected_countries = st.sidebar.multiselect("Select countries", df["country"].unique())
selected_date_range = st.sidebar.slider("Select date range", min_value=1000, max_value=12000, value=(5000,5000))

# Filter data by user input
filtered_df = df[(df["country"].isin(selected_countries)) & (df["date"].between(selected_date_range[0], selected_date_range[1]))]


# Display the filtered dataframe in the app
# st.write(filtered_df)



def plot_silhouette_score(X, k_min=2, k_max=(len(filtered_df)-1)):
    # Create a range of K values
    k_range = range(k_min, k_max+1)

    # Create an empty list to store the silhouette scores
    silhouette_scores = []
    
    # get top 3 silhoutte scores index

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
    st.pyplot(plt) # Show the plot in the app using st.pyplot()
    top_3= sorted(range(len(silhouette_scores)), key=lambda i: silhouette_scores[i], reverse=True)[:3]
    top_3= [(k_min+x) for x in top_3]
    return top_3
    


# Sidebar for user input
st.sidebar.title("Silhouette Score Plot")
plot_silhouette = st.sidebar.checkbox("Show", value=False)
top3=[]
if plot_silhouette:
    # Perform K-means clustering and plot the silhouette scores
    if len(filtered_df) > 3:
        X = filtered_df.iloc[:, 2:]
        if len(filtered_df) > 20:
            kmax=20
            top3=plot_silhouette_score(X, k_max=kmax) 
        else:
            top3=plot_silhouette_score(X)
    else:
        st.write("Error!!: Less than 3 samples' data available for selected countries and date range.")



st.sidebar.title("Number of Clusters:")
# select from top3 dropdown
n_clusters = st.sidebar.selectbox("Select number of clusters", top3)





def spectral_clustering(df3, k):
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df3)

    # Perform spectral clustering
    n_clusters = k  # Number of clusters
    clustering = SpectralClustering(
        n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors", random_state=42).fit(X)

    # create a dictionary with country as key and labels as value
    df_cont=filtered_df
    df_cont['date'] = df_cont['date'].astype(str)
    df_cont['countrydate'] = df_cont['country'] + '_' +df_cont['date']
    country_labels_S = dict(zip(df_cont.countrydate, clustering.labels_))
    labels=clustering.labels_
   

    # Visualize the clustering result
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
    
        
    ax.set_title("Spectral Clustering ({} clusters)".format(n_clusters))

    return fig, country_labels_S, labels

# Example usage in Streamlit app
st.title("Spectral Clustering")
country_labels={}
labels=[]
if n_clusters is not None:
    df3 = filtered_df
    fig, country_labels, labels = spectral_clustering(df3.iloc[:, 2:], n_clusters)
    st.pyplot(fig)







def plot_country_labels(country_labels):
    # Define the data for the Choropleth map
    data = go.Choropleth(
        locations=list(country_labels.keys()), # The countries
        z=list(country_labels.values()), # The cluster labels
        locationmode='country names',
        colorscale='Viridis', # The color scale to use
        colorbar=dict(title='Cluster'),
    )

    # Define the layout for the map
    layout = go.Layout(
        title='Cluster Labels by Country',
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
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



st.title('Country Clusters')
plot_country_labels(country_labels)





def umap_plot(df):
    # Apply UMAP to the dataset
    reducer = umap.UMAP()
    # df_umap = reducer.fit_transform(df)
    df_umap = umap.UMAP(min_dist=0.1, random_state=21).fit_transform(df)
    # size of the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Visualize the UMAP results
    plt.scatter(df_umap[:, 0], df_umap[:, 1], alpha=0.5)
    # color the points by their cluster assignment
    plt.scatter(df_umap[:, 0], df_umap[:, 1], c=labels, cmap='rainbow')

    df_cont=filtered_df
    df_cont['date'] = df_cont['date'].astype(str)
    df_cont['countrydate'] = df_cont['country'] + '_' +df_cont['date']
    annot2=df_cont['countrydate'].tolist()
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


umap_plot(filtered_df.iloc[:, 2:-1])