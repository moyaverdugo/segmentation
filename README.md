Segmentation

Notebooks structure:


i) Libraries used:

#Base
import numpy as np
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots as sp

#Data processing
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

# Principal Components
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Scaling
from sklearn.preprocessing import StandardScaler

# Segmentation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


ii) Data Wrangling

Drop nan
ORDERDATE column to datetime


1) Data Exploration

Validated the first and last date.
Validated total SALES amt.
Visualized total SALES amt per year.


2) Preparing dataframe for Segmentation


Feature Engineering NEWCUSTOMER

Aggreated dataframe by:
	 i)CUSTOMERNAME
	 ii)NEWCUSTOMER
	 aggregating:
	 i) SALES: mean
	 ii) QUANTITYORDERED: mean
	 iii) PRICEEACH: mean
	 iv) PRODUCTLINE: mode
	 v) DEALSIZE: mode
	added to the df:
	FREQUENCY and RECENCY by merging dataframes.
	Finally re-indexing

Later adding dummy variables for PRODUCTLINE and DEALSIZE.

Validating for colinearity.

Visualizing every the distribution of the data in every column.

Scaling.

Principal Components to visualize how the data as a whole distributes.

Visualizing all the data with SNS.

Transforming some columns with log_transformation.

Scaling.

PCA.

Visualizing (much better now).


3) SEGMENTATION

•Models tested: 1)K-means, 2)DBSCAN, 3)Agglomerative, 4) Gausian Mixture
•Result: K-means was the model chosen because it is the fastest given that all the models generated the same results.

The process for all the models was:

	Iterate with a for several segmentations so we can get enough data to determine the best hyperparameters.

	Ran the segmnetation with the chosen parameters

	Aggregate the data in the temp_df dataframe by segment and aggregating the mean.

	Visualize all the columns with Plotly library.



4) Export the segnmented data to CSV.