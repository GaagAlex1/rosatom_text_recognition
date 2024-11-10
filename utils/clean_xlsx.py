import pandas as pd

df = pd.read_excel('data.xlsx')

columns_to_clean = ['ДетальАртикул', 'ДетальНаименование', 'ЗаказНомер', 'СтанцияБлок']
for column in columns_to_clean:
    df[column] = df[column].str.replace('"', '', regex=False)

df["ЗаказНомер"] = df["ЗаказНомер"].str.strip()

output_path = 'cleaned_data.xlsx'
df.to_excel(output_path, index=False)
