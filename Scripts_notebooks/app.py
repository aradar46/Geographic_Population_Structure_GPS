
# import libraries
import streamlit as st
import pandas as pd
import Scripts.my_functions as mf

#? Set the page title
st.markdown("<h1 style='text-align: center; color: blue;'>GPSmaf</h1>", unsafe_allow_html=True)
#  a checkbox with bold font
st.markdown("<h3 style='text-align: center; color: black;'>A tool to visualize the geographic population structure of a set of samples based on their minor allele frequencies (MAF) of SNPs</h3>", unsafe_allow_html=True)
if st.checkbox('Check here to see the instructions and how to use the app'):
    st.write('''
    ## Instructions''')
    st.write('''
    This interface accepts a tab delimited csv.gz file with the following format:\n
    country\tdate\tSNP1\tSNP2\t...\n
    Sweden\t2000\tMAF1\tMAF2\t...\n
    ''')
    st.write(''' 
    1. Upload a csv.gz file or use the default test data.   
    2. Select countries and date range to filter the data. 
    3. Select number of clusters. ( You can run "Silhouette Score Plot" to find the best number of clusters)
    5. Run Clustering.\n
    GitHub repository: https://github.com/arash-darzian/Geographic_Population_Structure_GPS
    ''')
    
st.sidebar.title("1. Data")
container = st.sidebar.container()
with container:
    #! Allow user to upload a CSV file or load a default file
    uploaded_file = st.file_uploader("Choose a csv.gz file or work with default test data", type="gz")
    if uploaded_file is None:
        # st.info("No file uploaded. Loading default file...")
        df1 = pd.read_csv('Data/1_raw2freq/maf_filteered_test.csv.gz', sep="\t", compression='gzip')
        df1= pd.DataFrame(df1)
    else:
        df1 = pd.read_csv(uploaded_file, sep="\t", compression='gzip', header=None)
        df1= pd.DataFrame(df1)
st.write("---")

# protect the data
df=df1



#! Sidebar for user input
st.sidebar.title("2. Country and Date Selector")
select_all_countries = st.sidebar.checkbox("Select all countries", value=False)
if select_all_countries:
    selected_countries = df["country"].unique()
else:
    selected_countries = st.sidebar.multiselect("Select countries", df["country"].unique())
selected_date_range = st.sidebar.slider("Select date range", min_value=1000, max_value=12000, value=(5000,5000), step=1000)

# Filter data by user input
filtered_df = df[(df["country"].isin(selected_countries)) & (df["date"].between(selected_date_range[0], selected_date_range[1]))]

if len(filtered_df) > 0:
    st.sidebar.info(f"Number of avaialable samples after your filters: {len(filtered_df)}")



    