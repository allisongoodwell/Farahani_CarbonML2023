# Farahani_CarbonML2023
Contains python codes to reproduce results obtained in Farahani and Goodwell 2023 (submitted)

### Repository for Farahani et al, submitted Aug 2023
Python codes for “Information Flow Paths Characterize Drivers of Land-Atmosphere Carbon Fluxes from Machine Learning Models and Data” by: M. A. Farahani and A.E. Goodwell, submitted Aug 2023 to Water Resources Research

- **Dataset Preprocessing Script:** Readdata.py
The provided script is primarily designed for preprocessing and cleaning a set of CSV datasets. This script requires pandas and NumPy libraries for data manipulation and numerical operations respectively. The primary function accepts paths to five datasets. These datasets are categorized into two major groups: `Ne` and `Br`, (AMF_US.zip file) from 5 maize-soybean rotation sites in the FLUXNET2015 dataset:
    1. [USNe1](https://doi.org/10.17190/AMF/1246084)
    2. [USNe2](https://doi.org/10.17190/AMF/1246085)
    3. [USNe3](https://doi.org/10.17190/AMF/1246086)
    4. [USBr1](https://doi.org/10.17190/AMF/1246038)
    5. [USBr3](https://doi.org/10.17190/AMF/1246039)

    Note: There is meticulous attention to cleaning the dataset, ensuring that missing values and outliers don't skew analyses in further steps. It's crucial when using this script to ensure datasets provided match the expected format, and if there are any modifications to the data sources, appropriate changes are made to the script.

- **Information Theoretic Analysis:** Allfunctions.py

    This script offers a suite of functions that aim to compute various information theoretic quantities and normalize data: With these functions, users can:
1. Compute Shannon Entropy: shannon_entropy (x, bins): Computes the Shannon entropy of a dataset x using a specified number of bins.
2. Mutual Information Analysis: mutual_information (dfi, source, target, bins, reshuffle=0,ntests=0): Computes the mutual information between two columns of a dataframe dfi specified by source and target using a given bin size bins. It also provides an option for reshuffling the data to generate a null distribution and establish a critical threshold for mutual information.
3. Conditional Mutual Information: conditional_mutual_information (dfi, source, target, condition, bins, reshuffle=0): Computes the conditional mutual information between two columns of a dataframe dfi given a third column as a condition.
4. Information Partitioning: Decompose the information into unique, redundant, and synergistic components between two sources and a target variable.
5. interaction_information (mi_c, mi): Computes interaction information, a measure of how much one source variable (S1) influences the redundancy between a second source variable (S2) and a target.
6. normalized_source_dependency (mi_s1_s2, H_s1, H_s2): Computes a normalized version of mutual information between two source variables (S1 and S2) based on their individual entropies.
7. redundant_information_bounds (mi_s1_tar, mi_s2_tar, interaction_info): Provides bounds for redundancy based on the mutual information of the sources with the target and their interaction information.
8. rescaled_redundant_information (...): Computes a rescaled measure of redundancy between two source variables regarding a target variable.
9. information_partitioning (df, source_1, source_2, target, bins, reshuffle=0): Computes the total information, as well as the unique, redundant, and synergistic information between two source variables and a target variable.
10. Bin Number Calculation for Models: BinNumber_model (obs,model,bins_obs): Computes bin sizes for a model dataset based on observation data and bin sizes for the observation data.
11. Data Normalization: Normalized (frames, method): Normalize your data using one of the three methods: Standard, MinMax, or Quantile normalization.

- **Calculate all predictive and functional performance metrics:** Performance.py
    Load your data. Ensure that your data conforms to the labels and datasets specified. Utilize the defined functions to compute metrics and visualizations as per your requirements. The script imports two custom modules “allfunctions” (as af) and “Performance” (as per). Ensure these modules are present in the same directory or their paths are appropriately added.

    Note: It's advisable to run this on a machine with multiple cores as the script uses parallel processing for some computations. Ensure to have adequate memory, especially when dealing with large datasets, as some computations (like heatmaps) can be memory intensive.

- **Implantation of machine learning models:** multiple linear regression (MLR.py), random forest (RF.py) and Long Short-Term Memory (LSTM.py) models

    These scripts provide two main functions, “Model_l” (applies model on each dataset independently, local training) and “Model_r” (performs model on combined dataset, regional training). This script expects `frames` to be a dictionary where each key corresponds to an index in the `dataset` list and each value is a DataFrame containing the data for that site or condition. 

- **Main notebook:** Main.ipynb

    Main notebook to load the data, run ML models, save the result and calculate and plot all predictive and functional performance metrics. This Notebook imports all defined custom modules “readdata” (as rd), “allfunctions” (as af), “Performance” (as per), “MLR” (as mlr), “RF” (as rf), and “LSTM” (as lstm). Ensure these modules are present in the same directory or their paths are appropriately added.
    We also loaded the Goose Creek flux tower data directly in the main notebook (GCdata.zip file)

