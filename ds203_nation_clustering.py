import os
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
os.makedirs('data',exist_ok=True)
os.makedirs('figures',exist_ok=True)
indicator_dict={
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
raw_data_path='data/wdi_raw_2022.csv'
if os.path.exists(raw_data_path):
  df_raw=pd.read_csv(raw_data_path)
else:
  meta_url='https://api.worldbank.org/v2/country?format=json&per_page=400'
  meta_response=requests.get(meta_url)
  meta_json=meta_response.json()
  all_entities=meta_json[1]
  country_records=[]
  for item in all_entities:
    reg_val=item['region']['value']
    if reg_val!='Aggregates':
      code=item['id']
      name=item['name']
      inc=item['incomeLevel']['value']
      lend=item['lendingType']['value']
      record={
        'country_code':code,
        'country_name':name,
        'region':reg_val,
        'income_group':inc,
        'lending_type':lend
      }
      country_records.append(record)
  df_raw=pd.DataFrame(country_records)
  for ind_code,ind_name in indicator_dict.items():
    ind_url=f'https://api.worldbank.org/v2/country/all/indicator/{ind_code}?date=2022&format=json&per_page=400'
    ind_response=requests.get(ind_url)
    ind_json=ind_response.json()
    entries=ind_json[1]
    val_records=[]
    for ent in entries:
      c_iso=ent['countryiso3code']
      v=ent['value']
      val_records.append({'country_code':c_iso,ind_name:v})
    df_ind=pd.DataFrame(val_records)
    df_raw=pd.merge(df_raw,df_ind,on='country_code',how='left')
  df_raw.to_csv(raw_data_path,index=False)
raw_feature_names=list(indicator_dict.values())
print('Initial Dataset Information:')
print('Total countries loaded:',len(df_raw))
print('Total variables:',len(df_raw.columns))
print('\nMissing counts across indicators:')
print(df_raw[raw_feature_names].isnull().sum())
missing_counts_per_country=df_raw[raw_feature_names].isnull().sum(axis=1)
df_raw['missing_count']=missing_counts_per_country
threshold=2
df_clean=df_raw[df_raw['missing_count']<=threshold].copy()
df_clean=df_clean.reset_index(drop=True)
print(f'\nCountries retained with <= {threshold} missing values: {len(df_clean)}')
print('Dropped countries count:',len(df_raw)-len(df_clean))
df_clean['raw_gdp_per_capita']=df_clean['gdp_per_capita'].copy()
for feature_name in raw_feature_names:
  reg_median=df_clean.groupby('region')[feature_name].transform('median')
  df_clean[feature_name]=df_clean[feature_name].fillna(reg_median)
  glob_median=df_clean[feature_name].median()
  df_clean[feature_name]=df_clean[feature_name].fillna(glob_median)
print('Null count after regional imputation:',df_clean[raw_feature_names].isnull().sum().sum())
df_clean['log_gdp_per_capita']=np.log10(df_clean['gdp_per_capita'])

final_feature_names=[
  'log_gdp_per_capita',
  'gdp_growth',
  'inflation',
  'unemployment',
  'trade_pct_gdp',
  'capital_formation',
  'life_expectancy',
  'urban_pop_pct',
  'internet_users_pct',
  'electricity_access_pct',
  'secondary_school_enrollment'
]

display_feature_labels=[
  'Log GDP-PC',
  'GDP Growth',
  'Inflation',
  'Unemployment',
  'Trade % GDP',
  'Capital Formation',
  'Life Expectancy',
  'Urban Pop %',
  'Internet Users %',
  'Electricity Access %',
  'Secondary School %'
]

print('\nSummary Statistics of Cleaned Indicators:')
stat_summary=df_clean[final_feature_names].describe().T[['mean','std','50%','min','max']]
stat_summary['skewness']=[df_clean[col].skew() for col in final_feature_names]
print(stat_summary.round(2))

df_clean.to_csv('data/wdi_clean_2022.csv',index=False)

fig,axes=plt.subplots(2,4,figsize=(16,8))
axes_flat=axes.flatten()

sns.histplot(df_clean['raw_gdp_per_capita'],kde=True,ax=axes_flat[0],color='#1f77b4',bins=20)
axes_flat[0].set_title(f"Raw GDP per Capita (US$)\nSkewness: {df_clean['raw_gdp_per_capita'].skew():+.2f}")
axes_flat[0].set_xlabel('Current US$')

sns.histplot(df_clean['log_gdp_per_capita'],kde=True,ax=axes_flat[1],color='#2ca02c',bins=20)
axes_flat[1].set_title(f"Log10 GDP per Capita\nSkewness: {df_clean['log_gdp_per_capita'].skew():+.2f}")
axes_flat[1].set_xlabel('log10(US$)')

sns.histplot(df_clean['life_expectancy'],kde=True,ax=axes_flat[2],color='#d62728',bins=20)
axes_flat[2].set_title(f"Life Expectancy (Years)\nSkewness: {df_clean['life_expectancy'].skew():+.2f}")
axes_flat[2].set_xlabel('Years')

sns.histplot(df_clean['internet_users_pct'],kde=True,ax=axes_flat[3],color='#9467bd',bins=20)
axes_flat[3].set_title(f"Internet Users (% Pop)\nSkewness: {df_clean['internet_users_pct'].skew():+.2f}")
axes_flat[3].set_xlabel('% of Population')

sns.histplot(df_clean['electricity_access_pct'],kde=True,ax=axes_flat[4],color='#ff7f0e',bins=20)
axes_flat[4].set_title(f"Electricity Access (% Pop)\nSkewness: {df_clean['electricity_access_pct'].skew():+.2f}")
axes_flat[4].set_xlabel('% Access')

sns.histplot(df_clean['inflation'],kde=True,ax=axes_flat[5],color='#8c564b',bins=20)
axes_flat[5].set_title(f"Inflation Rate (%)\nSkewness: {df_clean['inflation'].skew():+.2f}")
axes_flat[5].set_xlabel('Annual CPI %')

sns.histplot(df_clean['unemployment'],kde=True,ax=axes_flat[6],bins=20)
axes_flat[6].set_title(f"Unemployment (%)\nSkewness: {df_clean['unemployment'].skew():+.2f}")
axes_flat[7].axis('off')

plt.tight_layout()
plt.savefig('figures/fig1_distributions.png',dpi=300)
plt.close()

income_levels=['Low income','Lower middle income','Upper middle income','High income']
fig,axes=plt.subplots(2,2,figsize=(13,9))

sns.boxplot(x='income_group',y='log_gdp_per_capita',data=df_clean,order=income_levels,ax=axes[0,0],hue='income_group',palette='Blues',legend=False)
axes[0,0].set_title('Log10 GDP per Capita across Income Groups')
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('log10(US$)')
axes[0,0].tick_params(axis='x',rotation=15)

sns.boxplot(x='income_group',y='life_expectancy',data=df_clean,order=income_levels,ax=axes[0,1],hue='income_group',palette='Reds',legend=False)
axes[0,1].set_title('Life Expectancy across Income Groups')
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Years')
axes[0,1].tick_params(axis='x',rotation=15)

sns.boxplot(x='income_group',y='internet_users_pct',data=df_clean,order=income_levels,ax=axes[1,0],hue='income_group',palette='Purples',legend=False)
axes[1,0].set_title('Internet Users across Income Groups')
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('% of Population')
axes[1,0].tick_params(axis='x',rotation=15)

sns.boxplot(x='income_group',y='electricity_access_pct',data=df_clean,order=income_levels,ax=axes[1,1],hue='income_group',palette='Oranges',legend=False)
axes[1,1].set_title('Electricity Access across Income Groups')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('% Access')
axes[1,1].tick_params(axis='x',rotation=15)

plt.tight_layout()
plt.savefig('figures/fig2_income_group_comparisons.png',dpi=300)
plt.close()

corr_data=df_clean[final_feature_names].corr()
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(corr_data,annot=True,fmt='.2f',cmap='coolwarm',vmin=-1,vmax=1,
  xticklabels=display_feature_labels,yticklabels=display_feature_labels,ax=ax)
ax.set_title('Pearson Correlation Matrix of Socioeconomic Indicators (2022)')
plt.xticks(rotation=45,ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/fig3_correlation_matrix.png',dpi=300)
plt.close()

fig,axes=plt.subplots(2,2,figsize=(14,10))
inc_palette={
  'Low income':'#d95f02',
  'Lower middle income':'#7570b3',
  'Upper middle income':'#1b9e77',
  'High income':'#e7298a'
}

sns.scatterplot(x='log_gdp_per_capita',y='life_expectancy',hue='income_group',
  hue_order=income_levels,palette=inc_palette,data=df_clean,s=65,ax=axes[0,0])
axes[0,0].set_title(f"Life Expectancy vs Log10 GDP per Capita (r = {df_clean['log_gdp_per_capita'].corr(df_clean['life_expectancy']):+.2f})")
axes[0,0].set_xlabel('log10(GDP per capita)')
axes[0,0].set_ylabel('Life Expectancy (Years)')

sns.scatterplot(x='log_gdp_per_capita',y='internet_users_pct',hue='income_group',
  hue_order=income_levels,palette=inc_palette,data=df_clean,s=65,ax=axes[0,1])
axes[0,1].set_title(f"Internet Users vs Log10 GDP per Capita (r = {df_clean['log_gdp_per_capita'].corr(df_clean['internet_users_pct']):+.2f})")
axes[0,1].set_xlabel('log10(GDP per capita)')
axes[0,1].set_ylabel('Internet Users (%)')

sns.scatterplot(x='secondary_school_enrollment',y='life_expectancy',hue='income_group',
  hue_order=income_levels,palette=inc_palette,data=df_clean,s=65,ax=axes[1,0])
axes[1,0].set_title(f"Life Expectancy vs Secondary Enrollment (r = {df_clean['secondary_school_enrollment'].corr(df_clean['life_expectancy']):+.2f})")
axes[1,0].set_xlabel('Secondary Gross Enrollment (%)')
axes[1,0].set_ylabel('Life Expectancy (Years)')

sns.scatterplot(x='log_gdp_per_capita',y='urban_pop_pct',hue='income_group',
  hue_order=income_levels,palette=inc_palette,data=df_clean,s=65,ax=axes[1,1])
axes[1,1].set_title(f"Urbanization vs Log10 GDP per Capita (r = {df_clean['log_gdp_per_capita'].corr(df_clean['urban_pop_pct']):+.2f})")
axes[1,1].set_xlabel('log10(GDP per capita)')
axes[1,1].set_ylabel('Urban Population (%)')

for ax_item in axes.flatten():
  ax_item.legend(title='Income Group',loc='lower right')

plt.tight_layout()
plt.savefig('figures/fig4_pairwise_scatters.png',dpi=300)
plt.close()

scaler=StandardScaler()
X_scaled=scaler.fit_transform(df_clean[final_feature_names])

pca_model=PCA()
X_pca=pca_model.fit_transform(X_scaled)
exp_variance_ratios=pca_model.explained_variance_ratio_
cum_variance_ratios=np.cumsum(exp_variance_ratios)

print('\nPCA Explained Variance Ratios:')
for idx,val in enumerate(exp_variance_ratios):
  print(f'PC{idx+1}: {val:.4f} (Cumulative: {cum_variance_ratios[idx]:.4f})')

fig,ax1=plt.subplots(figsize=(9,5))
comp_numbers=list(range(1,len(exp_variance_ratios)+1))
ax1.bar(comp_numbers,exp_variance_ratios*100,color='#3470a3',alpha=0.7)
ax1.set_xlabel('Principal Component Number')
ax1.set_ylabel('Explained Variance (%)',color='#3470a3')
ax1.set_xticks(comp_numbers)

ax2=ax1.twinx()
ax2.plot(comp_numbers,cum_variance_ratios*100,color='#d95f02',marker='o',linewidth=2)
ax2.axhline(y=70,color='gray',linestyle='--')
ax2.text(1.2,71,'70% Variance Threshold',color='gray')
ax2.set_ylabel('Cumulative Explained Variance (%)',color='#d95f02')
ax2.grid(False)

plt.title('PCA Scree Plot and Cumulative Variance')
fig.tight_layout()
plt.savefig('figures/fig5_pca_variance.png',dpi=300)
plt.close()

loadings_matrix=pd.DataFrame(
  pca_model.components_[:3].T,
  index=display_feature_labels,
  columns=[f'PC{i+1} ({exp_variance_ratios[i]:.1%})' for i in range(3)]
)
print('\nPCA Component Loadings for PC1, PC2, PC3:')
print(loadings_matrix.round(3))

fig,ax=plt.subplots(figsize=(8,7))
sns.heatmap(loadings_matrix,annot=True,fmt='.3f',cmap='vlag',center=0,ax=ax)
ax.set_title('PCA Component Loadings Matrix (Top 3 Components)')
plt.tight_layout()
plt.savefig('figures/fig6_pca_loadings.png',dpi=300)
plt.close()

k_candidates=list(range(2,9))
inertia_values=[]
print('\nK-Means search: k | Within-cluster sum of squares')
for k_val in k_candidates:
  km_trial=KMeans(n_clusters=k_val,random_state=42,n_init=20)
  km_trial.fit(X_scaled)
  inertia_values.append(km_trial.inertia_)
  print(k_val,round(km_trial.inertia_,2))

fig,ax=plt.subplots(figsize=(9,5))
ax.plot(k_candidates,inertia_values,marker='o',color='#1f77b4')
ax.axvline(x=4,color='#d62728',linestyle=':',label='Selected k=4')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Within-cluster sum of squares')
ax.set_xticks(k_candidates)
ax.legend()
ax.set_title('K-Means inertia across candidate cluster counts')
plt.tight_layout()
plt.savefig('figures/fig7_kmeans_selection.png',dpi=300)
plt.close()

final_k=4
kmeans_model=KMeans(n_clusters=final_k,random_state=42,n_init=30)
kmeans_model.fit(X_scaled)

df_clean['km_cluster']=kmeans_model.labels_

cluster_label_mapping={
  0:'Advanced development',
  1:'Lower development',
  2:'High inflation',
  3:'Intermediate development'
}
df_clean['cluster_title']=df_clean['km_cluster'].map(cluster_label_mapping)

df_clean['PC1']=X_pca[:,0]
df_clean['PC2']=X_pca[:,1]
df_clean['PC3']=X_pca[:,2]

fig,ax=plt.subplots(figsize=(11,7))
cluster_color_map={0:'#2b83ba',1:'#d7191c',2:'#fdae61',3:'#abdda4'}

sns.scatterplot(
  x='PC1',y='PC2',hue='km_cluster',palette=cluster_color_map,
  data=df_clean,s=80,alpha=0.9,ax=ax
)

landmark_countries=[
  'United States','Norway','Luxembourg','Lebanon','Zimbabwe',
  'Sudan','Central African Republic','Somalia','China','India',
  'Ukraine','Brazil'
]

for l_name in landmark_countries:
  match_rows=df_clean[df_clean['country_name']==l_name]
  if len(match_rows)>0:
    row_found=match_rows.iloc[0]
    ax.text(row_found['PC1']+0.12,row_found['PC2']+0.1,l_name,fontsize=8,weight='bold')

legend_elements,_=ax.get_legend_handles_labels()
ax.legend(handles=legend_elements,labels=[cluster_label_mapping[i] for i in range(4)],title='K-Means Cluster')
ax.set_title('2D PCA Projection of Discovered Socioeconomic Clusters (k=4)')
ax.set_xlabel(f'PC1 ({exp_variance_ratios[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({exp_variance_ratios[1]:.1%} variance)')
plt.tight_layout()
plt.savefig('figures/fig8_pca_clusters.png',dpi=300)
plt.close()

linkage_matrix=linkage(X_scaled,method='ward')
cut_distance=(linkage_matrix[-4,2]+linkage_matrix[-3,2])/2
fig,ax=plt.subplots(figsize=(14,6))
dendrogram(linkage_matrix,orientation='top',no_labels=True,color_threshold=cut_distance,ax=ax)
ax.axhline(y=cut_distance,color='black',linestyle='--',linewidth=1.5,label=f'Threshold for k=4: {cut_distance:.2f}')
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance)')
ax.set_xlabel(f'Countries/economies ({len(df_clean)} observations)')
ax.set_ylabel('Ward Dissimilarity Merge Distance')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/fig9_hierarchical_dendrogram.png',dpi=300)
plt.close()

hierarchical_model=AgglomerativeClustering(n_clusters=final_k,linkage='ward')
hierarchical_model.fit(X_scaled)
df_clean['agg_cluster']=hierarchical_model.labels_

fig,axes=plt.subplots(1,2,figsize=(14,5))

ct_km_hier=pd.crosstab(df_clean['km_cluster'],df_clean['agg_cluster'])
ct_km_hier.index=[cluster_label_mapping[i] for i in range(4)]
ct_km_hier.columns=[f'Ward {j}' for j in range(4)]

sns.heatmap(ct_km_hier,annot=True,fmt='d',cmap='Blues',ax=axes[0],cbar=False)
axes[0].set_title('K-Means vs Ward country counts')
axes[0].set_xlabel('Hierarchical Cluster')
axes[0].set_ylabel('K-Means Cluster')

ct_km_income=pd.crosstab(df_clean['km_cluster'],df_clean['income_group'])
ct_km_income.index=[cluster_label_mapping[i] for i in range(4)]
ct_km_income=ct_km_income[income_levels]


sns.heatmap(ct_km_income,annot=True,fmt='d',cmap='Greens',ax=axes[1],cbar=False)
axes[1].set_title('K-Means vs World Bank income group counts')
axes[1].set_xlabel('Held-out World Bank Income Group')
axes[1].set_ylabel('Discovered Cluster')

plt.tight_layout()
plt.savefig('figures/fig10_cluster_comparison.png',dpi=300)
plt.close()

cluster_centers=kmeans_model.cluster_centers_
dist_array=[]
for idx in range(len(df_clean)):
  assigned_c=df_clean['km_cluster'].iloc[idx]
  c_center=cluster_centers[assigned_c]
  point=X_scaled[idx]
  euc_dist=np.linalg.norm(point-c_center)
  dist_array.append(euc_dist)
df_clean['dist_to_centroid']=dist_array

fig,axes=plt.subplots(2,2,figsize=(14,9))
axes_flat_out=axes.flatten()

for c_num in range(4):
  sub_frame=df_clean[df_clean['km_cluster']==c_num]
  sub_sorted=sub_frame.sort_values(by='dist_to_centroid',ascending=False).head(5)
  sns.barplot(x='dist_to_centroid',y='country_name',data=sub_sorted,hue='country_name',palette='Blues_r',ax=axes_flat_out[c_num],legend=False)
  axes_flat_out[c_num].set_title(f'Cluster {c_num}: {cluster_label_mapping[c_num]}\nTop 5 Distant Countries')
  axes_flat_out[c_num].set_xlabel('Euclidean Distance to Centroid')
  axes_flat_out[c_num].set_ylabel('')

plt.tight_layout()
plt.savefig('figures/fig11_outlier_distances.png',dpi=300)
plt.close()

print('\nDiscovered Cluster Centroid Averages:')
mean_profiles=df_clean.groupby('km_cluster')[final_feature_names].mean().T
mean_profiles.columns=[cluster_label_mapping[i] for i in range(4)]
print(mean_profiles.round(2))

print('\nContingency Table (K-Means vs Income Groups):')
print(ct_km_income)

print('\nContingency Table (K-Means vs Geographic Regions):')
ct_regions=pd.crosstab(df_clean['km_cluster'],df_clean['region'])
ct_regions.index=[cluster_label_mapping[i] for i in range(4)]
print(ct_regions)

print('\nTop Unusual / Outlier Countries per Cluster:')
for c_num in range(4):
  print(f'\n--- Cluster {c_num}: {cluster_label_mapping[c_num]} ---')
  top_out=df_clean[df_clean['km_cluster']==c_num].sort_values(by='dist_to_centroid',ascending=False).head(3)
  for _,r in top_out.iterrows():
    c_n=r['country_name']
    c_inc=r['income_group']
    c_reg=r['region']
    c_d=r['dist_to_centroid']
    print(f'{c_n} ({c_inc}, {c_reg}): dist={c_d:.2f}')

df_clean.to_csv('data/wdi_final_results.csv',index=False)
print('\nPipeline completed successfully. All outputs and figures saved.')
