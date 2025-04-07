
1. **Input:** Users enter a list of PMIDs on the index page.
2. **Data Retrieval:** For each PMID, the application calls the NCBI e-utils API to retrieve associated GEO dataset IDs. For each GEO dataset, additional details (Title, Experiment type, Summary, Organism, Overall design) are fetched.
3. **TF-IDF & Clustering:** The text fields are combined and transformed using tf-idf. KMeans clustering groups similar datasets.
4. **Visualization:** PCA reduces the vectors to 2D and a scatter plot is generated. Each plotted point (a dataset) is annotated with its GEO ID.
5. **Display:** The clusters visualization and a table of GEO-to-PMID associations are shown on the results page.