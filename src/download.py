import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files("riondsilva21/hand-keypoint-dataset-26k", path="data", unzip=True)
