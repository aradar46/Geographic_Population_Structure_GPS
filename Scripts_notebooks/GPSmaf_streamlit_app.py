# import libraries
import streamlit as st
import pandas as pd
from umap import umap
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
# import matplotlib.pyplot as plt
import plotly.figure_factory as ff


# set page config
st.set_page_config(
    page_title="GPSmaf",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

################################################################################


# Title
st.title("GPSmaf")

# Subheader
st.subheader("Geographical Population Structure Infrence from Minor Allele Frequencies")

################################################################################
# functions
def umap_plot(df, data_label, colors_map, num_neigh):
    """Plot the UMAP projection of the data
    data_label is the name of each point
    colors_map is the color each point should have
    """
    # Fit UMAP and transform the data
    reducer = umap.UMAP(
        random_state=42,
        n_neighbors=num_neigh,
        min_dist=0.8,
        n_components=2,
        metric="euclidean",
    )
    df_umap = reducer.fit_transform(df)

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame(
        {
            "UMAP1": df_umap[:, 0],
            "UMAP2": df_umap[:, 1],
            "data_label": data_label,
            "colors_map": colors_map,
        }
    )

    # Get unique labels and assign colors
    unique_labels = df_plot["colors_map"].unique()
    num_labels = len(unique_labels)
    color_scale = px.colors.qualitative.Alphabet

    # Plot the UMAP projection
    fig = go.Figure()
    # size of figure
    fig.update_layout(width=600, height=800)
    for i, label in enumerate(unique_labels):
        label_data = df_plot[df_plot["colors_map"] == label]
        fig.add_trace(
            go.Scatter(
                x=label_data["UMAP1"],
                y=label_data["UMAP2"],
                mode="markers",
                marker=dict(
                    color=color_scale[i % len(color_scale)],
                ),
                text=label_data["data_label"],
                name=str(label),
            )
        )

    fig.update_traces(marker=dict(size=5), textfont=dict(size=8))
    fig.update_layout(
        title="UMAP projection",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        showlegend=True,
        plot_bgcolor="white",
    )
    # legend position below the plot
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="right", x=1)
    )

    #  # Reset the index if needed why? because the index is not continuous and it is not good for the dendrogram
    if isinstance(data_label, pd.Index):
        data_label = data_label.reset_index(drop=True)
    # Calculate distance between country-date points
    distances = umap.umap_.pairwise_distances(reducer.embedding_)
    # make a df of the distances and the data_label
    df_dist = pd.DataFrame(distances, index=data_label, columns=data_label)

    return df_dist, fig

################################################################################

def make_dendrogram(df_dist):
    distance_matrix = df_dist
    # Calculate linkage matrix for hierarchical clustering
    # linkage_matrix = linkage(distance_matrix.values, method='ward')

    # X = np.random.rand(15, 10) # 15 samples, with 10 dimensions each
    fig = ff.create_dendrogram(
        distance_matrix,
        color_threshold=5.5,
        labels=distance_matrix.index,
        orientation="left",
        linkagefun=lambda x: linkage(distance_matrix.values, method="ward"),
    )
    # x axis labels rotation
    fig.update_layout(xaxis_tickangle=-90)
    # y axis labels rotation
    # text size
    fig.update_layout(font=dict(size=6))

    # make points in the dendrogram clickable
    fig.update_traces(
        hovertemplate="Country: %{text}<extra></extra>",
        text=distance_matrix.index,
        selector=dict(type="scatter"),
    )
    # add text to the points in the dendrogram
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
    )
    fig.update_layout(width=600, height=800)
    fig.update_xaxes(showgrid=False, zeroline=True, showticklabels=True)
    # title
    fig.update_layout(title_text="Dendrogram", title_x=0.5)
    # remove borders
    fig.update_xaxes(showline=False, linewidth=0, linecolor="white")

    return fig

################################################################################

df = pd.read_csv("Data/Output/1_encoded_data_frame.csv.gz", compression="gzip")
dates = df.date.unique()
regions = df.region.unique()
countries = df.country.unique()


df = df[df.date.isin(dates)]
df = df[df.region.isin(regions)]
df = df[df.country.isin(countries)]


# select dates (in BP units)
dates = st.sidebar.multiselect("Select dates( BP * 1000 units):", dates)
df = df[df.date.isin(dates)]

# select countries
all_countries = st.sidebar.checkbox("Select all countries")
if all_countries:
    df = df[df.country.isin(countries)]
    st.sidebar.write("Number of groups: ", len(df))
else:
    countries = st.sidebar.multiselect("Select countries:", countries)
    df = df[df.country.isin(countries)]
    st.sidebar.write("Number of groups: ", len(df))


# select number of neighbors
mid_neg = int(np.sqrt(len(df)) - 1)
if mid_neg < 2:
    mid_neg = 2
num_neigh = st.sidebar.slider("Select number of neighbors:", 2, 40, mid_neg)

################################################################################
# sidebar button
if st.sidebar.button("Run"):
    col1, col2, col3 = st.columns([10, 1, 10])
    with col1:
        my_encoded_data = df.iloc[:, 7:].values
        # Plot
        # try:
        df_dist, fig = umap_plot(
            my_encoded_data,
            df["country"],
            df["region"].str.title(),
            num_neigh=num_neigh,
        )
        st.write(fig)
        # print number of points

    with col2:
        st.write("")

    with col3:
        # dendrogram
        dendo = make_dendrogram(df_dist)
        st.write(dendo)
        # except:
        #     st.write("Please select at least one region or country and one date")


################################################################################
