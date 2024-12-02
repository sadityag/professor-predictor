These are contents of this folder:

- There is 1 Jupyter notebook that includes the code for cleaning all of the collected data and putting them in a CSV files.   

- There are 2 folders: one for the Economic features and one for the # of faculty data obtained from NCES. For every set of data there are raw version and a cleaned version.

- There are 3 CSV files in this folder. 

  * data_description_and_units:

      This is a pandas data frame that has our features as columns. For each column, the first row describes the feature and the second row states the units. 

  * data_gapped: 
      This is a pandas data frame that has our features as columns. Each column will have NaN entries for years when there's no available data for this feature. 


  * data_interpolated: 
      This is a pandas data frame that has our features as columns. It is the same as data_gapped. However, for the 'faculty' column, we interpolated the missing data for a given year as the average of the pervious and following year's data.   