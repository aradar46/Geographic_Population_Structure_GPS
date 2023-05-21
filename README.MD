# GPSmaf v2.0

**Objective:**

Develop an interface that allows selecting populations and time. The program will merge patterns that are too close based on the minor allele frequencies of the chosen populations. The results will be printed on the screen.


**Description:**

This repository provides a pipeline for handling and analyzing ancient genetic data and infrence genetical relationship between samples through the ages. It covers various stages, including PLINK file parsing, missing data filtering, Minor Allele Frequency (MAF) calculation, clustering analysis, and dimensionality reduction using an ensemble of autoencoders. Additionally, it offers visualization tools for exploring the data in reduced-dimensional space and hierarchical clustering.



**Sections:**

- [Data Processing and Methodology](Notes/Methods_describtion.md): Describes the data processing and methodology.
- [Results](Notes/Results.md): Describes the results of the analysis.



**Data:**

- [Input Data: Plink files of Ancient DNA Samples ](Data/Input/description.txt)
- Output Data: `Data/Output`
  - `0_country_date_MAF.csv.gz`: contains the MAF for each SNP within each "country_date" group.
  - `1_encoded_data_frame.csv.gz`: contains the encoded data frame for each "country_date" group.


**Scripts:**

*Scripts_notebooks/*: contains the scripts and notebooks for the pipeline.

- `0_plink_to_countryDate_MAF.ipynb` : Parsing PLINK files and calculating MAF. Need near 500 GB of RAM to run.
- `1_clustring.ipynb` : Clustering analysis and visualization.

`GPSmaf_streamlit_app.py `: Streamlit app for the interface.


**Docker Implementation** 

```console
# Build the image
> docker build -t gps-app . #(the dot is important)
 
# Run the container with port forwarding
> docker run -itd --name gps -p 8501:8501 gps-image # (to run in detached mode)

# outputs the container id paste it in the next command
> docker attach <docked-id> (to attach to the container)
 
# Run Streamlit with port forwarding
> streamlit run --server.port 8501 app.py

#Click on the link below to open the app in your default browser
http://localhost:8501/
```

**Other Implementation** 

### install conda environment
  
  ```console
  # Create the environment from the environment.yml file
  conda env create -f conda_environment.yml

  # Activate the new environment
  conda activate gpsmaf

  # Verify that the new environment was installed correctly
  conda env list

```
### install pip requirements

inside the gpsmaf environment run the following command

  ```console
  # Create the environment from the environment.yml file
  pip install -r pip_requirements.txt

  # Verify that the new environment was installed correctly
  pip list
  ```

Start streamlit

  ```console
  streamlit run GPSmaf_streamlit_app.py
  ```


  