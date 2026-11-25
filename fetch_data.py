import requests
import pandas as pd
import json
indicators={
  'NY.GDP.PCAP.CD':'gdp_per_capita',
  'NY.GDP.MKTP.KD.ZG':'gdp_growth',
  'FP.CPI.TOTL.ZG':'inflation',
  'SL.UEM.TOTL.ZS':'unemployment',
  'NE.TRD.GNFS.ZS':'trade_pct_gdp',
  'NE.GDI.TOTL.ZS':'capital_formation',
  'SP.DYN.LE00.IN':'life_expectancy',
  'SP.URB.TOTL.IN.ZS':'urban_pop_pct',
  'IT.NET.USER.ZS':'internet_users_pct',
  'EG.ELC.ACCS.ZS':'electricity_access_pct',
  'SE.SEC.ENRR':'secondary_school_enrollment'
}
country_url='https://api.worldbank.org/v2/country?format=json&per_page=400'
res=requests.get(country_url).json()
countries_raw=res[1]
valid_countries={}
for c in countries_raw:
  reg=c['region']['value']
  if reg!='Aggregates':
    cid=c['id']
    valid_countries[cid]={
      'country_code':cid,
      'country_name':c['name'],
      'region':reg,
      'income_group':c['incomeLevel']['value'],
      'lending_type':c['lendingType']['value']
    }
df_meta=pd.DataFrame(list(valid_countries.values()))
print('Total real countries:',len(df_meta))
year=2022
indicator_dfs=[]
for code,name in indicators.items():
  url=f'https://api.worldbank.org/v2/country/all/indicator/{code}?date={year}&format=json&per_page=400'
  r=requests.get(url).json()
  entries=r[1]
  records=[]
  for e in entries:
    cid=e['countryiso3code']
    val=e['value']
    if cid in valid_countries:
      records.append({'country_code':cid,name:val})
  df_ind=pd.DataFrame(records)
  indicator_dfs.append(df_ind)
df_merged=df_meta.copy()
for d in indicator_dfs:
  df_merged=pd.merge(df_merged,d,on='country_code',how='left')
print('Merged shape:',df_merged.shape)
print('Missing values per indicator:')
print(df_merged[list(indicators.values())].isnull().sum())
df_merged.to_csv('data/wdi_raw_2022.csv',index=False)
print('Saved to data/wdi_raw_2022.csv')
