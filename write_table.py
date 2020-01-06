import pandas as pd
from pytablewriter import MarkdownTableWriter

data = pd.read_json('result/results.json')
writer = MarkdownTableWriter()
writer.from_dataframe(data.T, add_index_column=True)
writer.write_table()
