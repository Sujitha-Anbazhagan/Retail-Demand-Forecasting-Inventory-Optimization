import kagglehub

path = kagglehub.competition_download(
    "m5-forecasting-accuracy"
)

print("Dataset downloaded to:")
print(path)