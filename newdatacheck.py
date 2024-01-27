import pandas as pd
import matplotlib.pyplot as plt

file_path = 'UNPC.en-ru.en'
file_path2 = 'UNPC.en-ru.ru'

train_file = pd.read_csv(file_path, sep = ',')

file_path.head()
file_path2.head()