

import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk

data = pd.read_csv("Data/DataS1_freq_labels (curated).csv", sep="\t")



# Define labels and their colors
labels = data["label"].unique()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9b59b6", "#3498db"]
label_color_dict = {label: color for label, color in zip(labels, colors)}

# Sidebar for user input
st.sidebar.title("Country and Date Selector")
select_all_countries = st.sidebar.checkbox("Select all countries", value=False)
if select_all_countries:
    selected_countries = data["country"].unique()
else:
    selected_countries = st.sidebar.multiselect("Select countries", data["country"].unique())
selected_date_range = st.sidebar.slider("Select date range", min_value=1, max_value=10, value=(5, 7))

# Filter data by user input
filtered_data = data[(data["country"].isin(selected_countries)) & (data["date"].between(selected_date_range[0], selected_date_range[1]))]

# Create plot
if len(filtered_data) > 0:
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X("label:N", sort=list(labels)),
        y=alt.Y("country:N", sort=alt.SortField(field="country", order="ascending")),
        color=alt.Color("label:N", scale=alt.Scale(domain=list(label_color_dict.keys()), range=list(label_color_dict.values()))),
        order=alt.Order("country", sort="ascending")
    ).properties(
        width=500,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.write("No data available for selected countries and date range.")


# Create map
# st.title("Map")
# st.map(filtered_data)


# # Create map
# view_state = pdk.ViewState(latitude=filtered_data["latitude"].iloc[0], 
#                            longitude=filtered_data["longitude"].iloc[0], 
#                            zoom=2, bearing=0, pitch=0)
# marker_layer = pdk.Layer("ScatterplotLayer", 
#                           data=filtered_data, 
#                           get_position="[longitude, latitude]",
#                           get_radius=100000,
#                           get_color=[255, 0, 0])
# map = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", 
#                initial_view_state=view_state, 
#                layers=[marker_layer])

# # Display map
# st.pydeck_chart(map)

if len(filtered_data) > 0:
    # Create map
    view_state = pdk.ViewState(latitude=filtered_data["latitude"].iloc[0], 
                            longitude=filtered_data["longitude"].iloc[0], 
                            zoom=2, bearing=0, pitch=0)
    marker_layer = pdk.Layer("ScatterplotLayer",
                            data=filtered_data, 
                            get_position="[longitude, latitude]",
                            get_radius=100000,
                            get_fill_color= [255, 0, 0, 44],
                            get_line_color=[22, 22, 22, 20],
                            pickable=True,
                            auto_highlight=True)
    marker_layer = pdk.Layer("ScatterplotLayer", 
                            data=filtered_data, 
                            get_position="[longitude, latitude]",
                            get_radius=100000,
                            get_fill_color= [255, 0, 0, 44],
                            get_line_color=[22, 22, 22, 20],
                            pickable=True,
                        auto_highlight=True)
    map = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", 
                initial_view_state=view_state, 
                layers=[marker_layer])

    # Display map
    st.pydeck_chart(map)


