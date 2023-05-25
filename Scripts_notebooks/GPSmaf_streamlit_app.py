# import libraries
import streamlit as st
import pandas as pd
import umap
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

# Description
st.write('How to use this app:')
st.write('1. Select the dates you want to include in the analysis')
st.write('2. Select the countries you want to include in the analysis. If you select all countries, the analysis will be done for all countries')
st.write('3. Select the number of neighbors for the UMAP projection. The default will be calculated automatically, based on the number of country-date groups selected')
st.write('4. Run the analysis and the UMAP projection and dendrogram will be shown. You can hover over the points to see the country and date of each point')

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
            textposition="top center",  # Show labels without hover
            name=str(label),
        )
    )
    # make points bigger
    
    #make background black
    fig.update_yaxes(
        # mirror=True,
        ticks='outside',
        # showline=True,
        linecolor='white',
        gridcolor='white'
                    )


    fig.update_traces(marker=dict(size=8), textfont=dict(size=8))
    fig.update_layout(
        title="UMAP projection", title_x=0.5,
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        showlegend=True,
        # plot_bgcolor="#101010",

        
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
    fig = ff.create_dendrogram(
        distance_matrix,
        color_threshold=5.5,
        labels=distance_matrix.index,
        orientation="left",
        linkagefun=lambda x: linkage(distance_matrix.values, method="ward"),
    )
    
    fig.update_layout(xaxis_tickangle=-90,
        font=dict(size=4),
        hoverlabel=dict( font_size=16, font_family="Rockwell"),
        width=600, height=800,
        title_text="Dendrogram",
        title_x=0.5,
        )

    fig.update_traces(
        hovertemplate="Country: %{text}<extra></extra>",
        text=distance_matrix.index,
        selector=dict(type="scatter"),
    )


    # fig.update_xaxes(showline=False, linewidth=0, linecolor="white")
    # fig.update_yaxes(tickfont=dict(size=6))

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
tool_tip_umap='The "number of neighbors" parameter in UMAP refers to the number of neighboring points considered when constructing the topological structure of the data.'
mid_neg = int(np.sqrt(len(df)) - 1)
if mid_neg < 2:
    mid_neg = 2
num_neigh = st.sidebar.slider("Select number of neighbors:", 2, 40, mid_neg, help=tool_tip_umap)


# chose between country_region 	country_date	country 
# Using tooltip with a button
tooltip1 = "Label which will be shown in the plot and dendrogram."
group_by = 'country'
# if user hover over the question mark, show the following text

if len(dates)>1:
    group_by = "country_date"



################################################################################
# sidebar button
if st.sidebar.button("Run"):
    if len(df)<5:
        st.header("There is not enough data to run the analysis. Please select more countries and dates.")
    else:
        try:
            col1, col2, col3 = st.columns([10, 1, 10])
            with col1:
                my_encoded_data = df.iloc[:, 7:].values
                # Plot
                # try:
                df_dist, fig = umap_plot(
                    my_encoded_data,
                    df[group_by],
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
        except:
            st.header("Please select at least one country and one date. The more better.")
 

################################################################################



# write a alert
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write("Some tips:")
st.sidebar.write("--> If the points are scattered, try to select smaller number of neighbors")
st.sidebar.write("--> The plots are interactive, you can zoom in and out and save the plots")
st.sidebar.write("--> If you select more than one date, the labels will be shown as < countryName + _ + int(bp_date/1000) >")



# github link
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")

# hyperlink
# 
st.sidebar.markdown(
    """<a href="https://github.com/arash-darzian/Geographic_Population_Structure_GPS" target="_blank">GitHub Repository</a>""",
    unsafe_allow_html=True,
)