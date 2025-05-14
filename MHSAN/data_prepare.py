import kagglehub

# 指定你想要下載到的目錄
download_dir = "./"

# 下載資料集到指定的路徑
path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset", download_dir=download_dir)

print("資料集下載路徑：", path)
