
# import libraries
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
import my_functions as mf
import graphviz


container = st.container()
with container:
    #? Set the page title
    st.title("Geographic Population Structure (GPS) Based on Minor Allele Frequencies (MAF) of SNPs")
    st.info('This interface accepts a tab delimited csv.gz file with the following format:')
    st.text('country\tdate\tSNP1\tSNP2\t...\nSweden\t2000\tMAF1\tMAF2\t...')
    st.info('github link:')
    #! Allow user to upload a CSV file or load a default file
    uploaded_file = st.file_uploader("Choose a csv.gz file work with default data", type="gz")

    # download default data
    if st.button("Download Test Data"):
        st.download_button(
            label="Download",
            data='Data/1_raw2freq/test_1.csv.gz',
            file_name="Data/1_raw2freq/test_1.csv.gz",)
        
    if uploaded_file is None:
        # st.info("No file uploaded. Loading default file...")
        df1 = pd.read_csv('Data/1_raw2freq/test_1.csv.gz', sep="\t", compression='gzip')
        df1= pd.DataFrame(df1)
    else:
        df1 = pd.read_csv(uploaded_file, sep="\t", compression='gzip', header=None)
        df1= pd.DataFrame(df1)
        
    # Display the dataframe in the app
    st.write(df1.head(5))
    

st.write("---")

# protect the data
df=df1



#! Sidebar for user input
st.sidebar.title("Country and Date Selector")
select_all_countries = st.sidebar.checkbox("Select all countries", value=False)
if select_all_countries:
    selected_countries = df["country"].unique()
else:
    selected_countries = st.sidebar.multiselect("Select countries", df["country"].unique())
selected_date_range = st.sidebar.slider("Select date range", min_value=1000, max_value=12000, value=(5000,5000), step=1000)

# Filter data by user input
filtered_df = df[(df["country"].isin(selected_countries)) & (df["date"].between(selected_date_range[0], selected_date_range[1]))]




top3=[]
n_clusters=0
if len(filtered_df) > 3:
    #! Sidebar for user number of clusters
    st.sidebar.title("Spectral Clustring")
    st.sidebar.subheader("Number of Clusters:")
    # sidebar button Silhouette Score Plot
    if st.sidebar.button("Run Silhouette Score Plot"):
    # plot the silhouette scores
        if len(filtered_df) > 3:
            X = filtered_df.iloc[:, 2:]
            if len(filtered_df) > 20:
                kmax=20
                top3=mf.plot_silhouette_score(X, k_max=kmax) 
                st.sidebar.info(f"Top 3 Silhouette Scores are cluster numbers of: {top3}")
            else:
                kmax=len(filtered_df)-1
                top3=mf.plot_silhouette_score(X, k_max=kmax)
                st.sidebar.info(f"Top 3 Silhouette Scores: {top3}")
        else:
            st.sidebar.info(f"Top 3 Silhouette Scores are cluster numbers of: {top3}")
    min_value = 2
    max_value = len(filtered_df)-1
    n_clusters = st.sidebar.slider('Select a number clusters', min_value, max_value)





# create a dictionary with country as key and labels as value
df_cont=filtered_df
df_cont['date'] = df_cont['date'].astype(str)
df_cont['countrydate'] = df_cont['country'] + '_' +df_cont['date']
country_labels={}
labels=[]

if n_clusters>1:
    if st.sidebar.button("Run Clustering"):
        st.title("Spectral Clustering")
        country_labels, labels = mf.spectral_clustering(filtered_df.iloc[:, 2:-1], n_clusters, df_cont.countrydate)
        # st.pyplot(fig)
        st.subheader('Clusters on Map')
        mf.plot_country_labels(country_labels)
        st.subheader('Clusters by Details')
        # Create a graphlib graph object
        graph = graphviz.Digraph()
        # Loop through the rows of the DataFrame and add edges to the graph
        for k, v in country_labels.items():
            graph.edge(str(k), str(v))
        # unflatten the graph
        graph.graph_attr['rankdir'] = 'LR'
        st.graphviz_chart(graph)
        

if len(filtered_df)>3:
    st.sidebar.title("UMAP & TSNE")
    if st.sidebar.button("Run t-SNE & UMAP "):
            s1, s2 = st.columns(2)
            with s1:
                st.subheader("t-SNE Plot")
                mf.tsne_plot(filtered_df.iloc[:, 2:-1], df_cont.countrydate)
            with s2:
                st.subheader('UMAP Plot')
                mf.umap_plot(filtered_df.iloc[:, 2:-1], df_cont.countrydate)
        
    