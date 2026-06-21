import kagglehub

# Download latest version
## This downloads to computer files
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print("Path to dataset files:", path)


