import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

desktop = Path.home() / 'Desktop'
data_wd = desktop / 'nba_drafts'

filename = data_wd / '2021.csv'
df = pd.read_csv(filename)

df.columns