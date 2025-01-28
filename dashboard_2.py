import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st. set_page_config(layout="wide")

# Load Data -> .csvs from load_databases.py
df_raw = pd.read_csv('df_database.csv')
df2 = pd.read_csv('df2_database.csv')
    
## Sort based on category for plotting
df_raw = df_raw.sort_values(by = 'CatID')

### Define mappings
## Category to Category ID
cat_dict1 = {
   'ES_M': [601,701,703,801,803],  
   'MS_M': [103,501,502,503,000],
   'HS_M': [101,201,202,301,302],
   'ES_F': [602,702,704,802,804],
   'MS_F': [104,402,000,000,000],
   'HS_F': [102,203,401,000,000],
   }
# Convert dict to dataframe for indexing
cat_df1 = pd.DataFrame(cat_dict1)

cat_dict2 = {
   'M': [101,201,202,301,302,103,501,502,503,601,701,703,801,803],  
   'F': [102,203,401,104,402,602,702,704,802,804,000,000,000,000],
   }
# Convert dict to dataframe for indexing
cat_df2 = pd.DataFrame(cat_dict2)

cat_df3 = [101,201,202,301,302,103,501,502,503,601,701,703,801,803,102,203,401,104,402,602,702,704,802,804]

cat_grd = {
    101: 12,
    202: 12,
    201: 10, 
    301: 12,
    302: 9,
    103: 8,
    501: 8,
    502: 7,
    503: 6,
    601: 5,
    701: 5,
    703: 4,
    801: 3,
    803: 1,
    102: 12,
    203: 12,
    401: 12,
    104: 8,
    402: 8,
    602: 5,
    702: 5,
    704: 4,
    802: 3,
    804: 1,    
    }

cat_adds = {
    511: 'Intermediate 7-8th Grade Male',
    513: 'Intermediate 6th Grade Male',
    521: 'Novice 7-8th Grade Male',
    523: 'Novice 6th Grade Male',
    412: 'Intermediate 6-8th Grade Female',
    422: 'Novice 6-8th Grade Female'
    }


## Race order based on venue name, for indexing
race_ord = ['Merrell','Addison','Owaippe','Maybury','Stony','Heritage']

## Course order, for indexing
crs_ord = ['Advanced', 'Intermediate', 'Beginner']

## Color palette for coloring based on number of laps for the category
lap_palette = {
    '1 Beginner': 'xkcd:blush',
    '2 Beginner': 'xkcd:blood orange',
    '3 Beginner': 'xkcd:brick orange',
    '1 Intermediate': 'royalblue',
    '2 Intermediate': 'blue',
    '1 Advanced': 'violet',
    '2 Advanced': 'mediumorchid',
    '3 Advanced': 'purple',
    }

key_palette = {
    'GRADUATED': 'lightgrey',
    'MANDATORY PERFORMANCE': 'red',
    'SUGGESTED': 'deeppink',
    'MAXED OUT': 'maroon',
    'GRADE BASED': 'dodgerblue',
    'NEW NOVICE': 'darkviolet',
    'NONE': 'grey'
    }


marker_palette = {
    'GRADUATED': 'X',
    'MANDATORY PERFORMANCE': '^',
    'SUGGESTED': 'd',
    'MAXED OUT': '>',
    'GRADE BASED': 's',
    'NEW NOVICE': 'v',
    'NONE': 'o'
    }

cat_opts = ['Varsity Male',
            'Junior Varsity 11-12th Grade Male',
            'Junior Varsity 9-10th Grade Male',
            'Novice 10-12th Grade Male',
            'Novice 9th Grade Male',
            'Advanced Middle School Male',
            'Novice 8th Grade Male',
            'Novice 7th Grade Male',
            'Novice 6th Grade Male',
            'Advanced Elementary Male',
            'Elementary 5th Grade Male',
            'Elementary 4th Grade Male',
            'Elementary 2nd-3rd Grade Male',
            'Elementary PreK-1st Grade Male',
            'Varsity Female',
            'Junior Varsity Female',
            'Novice 9-12th Grade Female',
            'Advanced Middle School Female',
            'Novice 6-8th Grade Female',
            'Advanced Elementary Female',
            'Elementary 5th Grade Female',
            'Elementary 4th Grade Female',
            'Elementary 2nd-3rd Grade Female',
            'Elementary PreK-1st Grade Female'
            ]   

plot_sel_1 = st.sidebar.selectbox(
    "Plot Type",
    ["by Venue", "Multiple Venues", "Series Ranking"]
    )

if plot_sel_1 == "Series Ranking":
    x_sel_1 = st.sidebar.radio(
        "X-Axis",
        ['SeriesRank','SeriesPoints']
        )
else:
    x_sel_1 = st.sidebar.radio(
        "X-Axis",
        ['LapAvg','Final']
        )

cat_sel_1 = st.sidebar.multiselect(
    "Categories",
    cat_opts,
    )

if plot_sel_1 == "by Venue":
    race_sel_1 = st.sidebar.selectbox(
        "Race Venue",
        race_ord
        )
    st.sidebar.write("--UPGRADES--")
    mand_upgrd_6 = st.sidebar.slider(
        "Mandatory Upgrade if above this Rank in a field of 6-10:",
        min_value = 1,
        max_value = 10,
        step = 1,
        )
    mand_upgrd_11 = st.sidebar.slider(
        "Mandatory Upgrade if above this Rank in a field of >11:",
        min_value = 3,
        max_value = 15,
        step = 1,
        )
    sugg_upgrd = st.sidebar.slider(
        "Suggested Upgrade if above this Rank in a field of >11:",
        min_value = 0,
        max_value = 20,
        step = 1,
        )
    upgrd_thresh = st.sidebar.slider(
        "....for this many races:",
        min_value = 2,
        max_value = 6,
        step = 1)
    st.sidebar.write("--NOVICE DOWNGRADES--")
    nov_new_pct = st.sidebar.slider(
        "Novice downgrade if above this Rank Percentile:",
        min_value = 30,
        max_value = 90,
        step = 1)
    nov_new_thresh = st.sidebar.slider(
        "...for this many races:",
        min_value = 1,
        max_value = 6,
        step = 1)
elif plot_sel_1 == "Multiple Venues":
    race_sel_1 = st.sidebar.multiselect(
        "Race Venue (multi)",
        race_ord
        )
else: 
    race_sel_1 = race_ord
    st.sidebar.write("--UPGRADES--")
    mand_upgrd = st.sidebar.slider(
        "Mandatory Upgrade below this Rank Percentile (0 = MiSCA rules):",
        min_value = 0,
        max_value = 40,
        step = 1,
        )
    sugg_upgrd = st.sidebar.slider(
        "Suggested Upgrade below this Rank Percentile:",
        min_value = 0,
        max_value = 40,
        step = 1,
        )
    upgrd_thresh = st.sidebar.slider(
        "....for this many races:",
        min_value = 1,
        max_value = 6,
        step = 1)
    st.sidebar.write("--NOVICE DOWNGRADES--")
    nov_new_pct = st.sidebar.slider(
        "Novice downgrade if above this Rank Percentile:",
        min_value = 30,
        max_value = 90,
        step = 1)
    nov_new_thresh = st.sidebar.slider(
        "...for this many races:",
        min_value = 1,
        max_value = 6,
        step = 1)


# Start determining upgrades 
j = 0
while j < 6:
    k=0
    cat_cnts = df2['CatID_'+str(j+1)].value_counts()
    pct_col_hdr = 'RankPct_'+str(j+1)
    cat_col_hdr = 'CatID_'+str(j+1)
    rnk_col_hdr = 'Rank_'+str(j+1)
    idx = df2.columns.get_loc(rnk_col_hdr)+1
    df2.insert(idx,pct_col_hdr,np.nan)
    while k < len(cat_df3):
        mask = (df2[cat_col_hdr] == cat_df3[k])
        df2_mask = df2[mask]
        df2.loc[mask,pct_col_hdr] = (df2_mask[rnk_col_hdr]-1) / (cat_cnts[cat_df3[k]])*100
        if j == 0:
            mask = (df2['CatID'] == cat_df3[k])
            df2_mask = df2[mask]
            df2.loc[mask,'Pct_Series'] = (df2_mask['SeriesRank']-1) / (cat_cnts[cat_df3[k]])*100
        k += 1
    j += 1
    
df2['Grade_25'] = df2['Grade']+1   
n = 0
while n < df2.shape[0]:
    catid = df2.loc[n,'CatID']
    l = 0
    while l < 6:
        if catid != df2.loc[n,'CatID_'+str(l+1)]:
            df2.loc[n,'Rank_'+str(l+1)] = np.nan
        l += 1
    if not(catid == 0 or np.isnan(catid) or df2.loc[n,'Grade_25'] > 12):
        idx = list(cat_grd.keys()).index(catid)
        while list(cat_grd.values())[idx] < df2.loc[n,'Grade_25']:
        #    print(f"{df2.loc[n,'RegID']} // {df2.loc[n,'Grade_25']} // {list(cat_grd.values())[idx]} // {idx}")
            idx -= 1
        #print(f"{df2.loc[n,'RegID']} // {df2.loc[n,'Grade_25']} // {list(cat_grd.values())[idx]} // {idx}")
        df2.loc[n,'CatID_25_dflt'] = list(cat_grd.keys())[idx]
    n += 1

cat_cnts = df2['CatID'].value_counts()
n = 0
while n < df2.shape[0]:
    catid = df2.loc[n,'CatID']
    grade = df2.loc[n,'Grade']
    l = 0
    tmp_upgrd = 0
    tmp_upgrd_2 = 0
    tmp_upgrd_3 = 0
    new_nov = 0
    num_start = 0
    while l < 6:
        # MiSCA upgrade method
        if (cat_cnts[catid] >= 6 and cat_cnts[catid] < 11 and df2.loc[n,'Rank_'+str(l+1)] <= mand_upgrd_6) or (cat_cnts[catid] >=11 and df2.loc[n,'Rank_'+str(l+1)] <= mand_upgrd_11):
            tmp_upgrd += 1
        if (cat_cnts[catid] >=11 and df2.loc[n,'Rank_'+str(l+1)] <= sugg_upgrd):
            tmp_upgrd_2 += 1
        if catid in [701, 702, 502, 503, 402] and not np.isnan(df2.loc[n,'Rank_'+str(l+1)]):
            num_start += 1
            if df2.loc[n, 'RankPct_'+str(l+1)] >= nov_new_pct:
                new_nov += 1
        l += 1
    if df2.loc[n,'Grade_25'] > 12:
            df2.loc[n,'Upgrd_1'] = 'GRADUATED'
            df2.loc[n,'CatID_25_Upgrd_1'] = df2.loc[n,'CatID_25_dflt']
    elif not(catid == 0 or np.isnan(catid)):
        if tmp_upgrd >= 2:
            if grade in [5] and catid in [601]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 103
            elif grade in [5] and catid in [602]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 104
            elif catid in [502, 503]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 103
            elif grade in [6,7] and catid in [402]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 104
            elif grade in [8] and catid in [501]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 201
            elif grade in [8] and catid in [402]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 203
            elif catid in [302]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 201
            elif grade in [10, 11] and catid in [301]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 202
            elif grade in [9, 10, 11] and catid in [401]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 203
            elif grade in [9, 10, 11] and catid in [201, 202]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 101
            elif grade in [9, 10, 11] and catid in [401]:
                df2.loc[n,'Upgrd_1'] = 'MANDATORY PERFORMANCE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 102
            else:
                df2.loc[n,'Upgrd_1'] = 'MAXED OUT'
                df2.loc[n,'CatID_25_Upgrd_1'] = df2.loc[n,'CatID_25_dflt']
        elif (sugg_upgrd > 0 and tmp_upgrd_2 >= upgrd_thresh):
            if grade in [5] and catid in [601]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 103
            elif grade in [5] and catid in [602]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 104
            elif catid in [502, 503]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 103
            elif grade in [6,7] and catid in [402]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 104
            elif grade in [8] and catid in [501]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 201
            elif grade in [8] and catid in [402]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 203
            elif catid in [302]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 201
            elif grade in [10, 11] and catid in [301]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 202
            elif grade in [9, 10, 11] and catid in [401]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 203
            elif grade in [9, 10, 11] and catid in [201, 202]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 101
            elif grade in [9, 10, 11] and catid in [401]:
                df2.loc[n,'Upgrd_1'] = 'SUGGESTED'
                df2.loc[n,'CatID_25_Upgrd_1'] = 102
            else:
                df2.loc[n,'Upgrd_1'] = 'MAXED OUT'
                df2.loc[n,'CatID_25_Upgrd_1'] = df2.loc[n,'CatID_25_dflt']
        elif num_start == new_nov or new_nov >= nov_new_thresh:
            if catid in [502, 503]:
                df2.loc[n,'Upgrd_1'] = 'NEW NOVICE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 521
            elif catid in [701, 503]:
                df2.loc[n,'Upgrd_1'] = 'NEW NOVICE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 523
            elif catid in [702, 402]:
                df2.loc[n,'Upgrd_1'] = 'NEW NOVICE'
                df2.loc[n,'CatID_25_Upgrd_1'] = 412
        elif catid != df2.loc[n,'CatID_25_dflt']:
            df2.loc[n,'Upgrd_1'] = 'GRADE BASED'
            df2.loc[n,'CatID_25_Upgrd_1'] = df2.loc[n,'CatID_25_dflt']
        else:
            df2.loc[n,'Upgrd_1'] = 'NONE'
            df2.loc[n,'CatID_25_Upgrd_1'] = df2.loc[n,'CatID_25_dflt']
    n += 1    

df2.to_csv('df2.csv', index=False)



## Create key column for coloring
df_raw['key_1'] = df_raw['SeriesRank'].apply(lambda x: 1 if x <= 5 else 2 if x <= 10 else 0)
df_raw['Laps_Course'] = df_raw['Laps'].astype(str) + ' ' + df_raw['Course']

df_raw = df_raw.merge(df2[['RegID','Upgrd_1']], on = 'RegID', how = 'left')

cat_id = []
cat_list = list(cat_grd.keys())
for var in cat_sel_1:
    temp_idx = cat_opts.index(var)
    cat_id.append(cat_list[temp_idx])
    
df_tmp = df_raw.loc[df_raw['CatID'].isin(cat_id)]
if type(race_sel_1) == str: 
    df = df_tmp.loc[df_tmp['Venue'] == race_sel_1]
else:
    df = df_tmp.loc[df_tmp['Venue'].isin(race_sel_1)]
    
if plot_sel_1 == "by Venue":
    fig = plt.figure(figsize=(14, 10))
    if df.shape[0] > 0:
        # sns.swarmplot(data = df, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, size = 4, dodge = False, legend = True)
        df_tmp = df[(df['Upgrd_1'] == 'GRADUATED')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 'X', size = 5, dodge = False, legend = False)
        df_tmp = df[(df['Upgrd_1'] == 'MANDATORY PERFORMANCE')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = '^', size = 6, dodge = False, legend = False) 
        df_tmp = df[(df['Upgrd_1'] == 'SUGGESTED')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 'd', size = 6, dodge = False, legend = False) 
        df_tmp = df[(df['Upgrd_1'] == 'MAXED OUT')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = '>', size = 6, dodge = False, legend = False)  
        df_tmp = df[(df['Upgrd_1'] == 'GRADE BASED')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 's', size = 4, dodge = False, legend = False)
        df_tmp = df[(df['Upgrd_1'] == 'NEW NOVICE')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 'v', size = 5, dodge = False, legend = False) 
        df_tmp = df[(df['Upgrd_1'] == 'NONE')]
        sns.swarmplot(data = df_tmp, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 'o', size = 4, dodge = False, legend = True)
        
        sns.boxplot(data = df, x = x_sel_1, y = 'Cat', hue = 'Laps_Course', palette = lap_palette, linewidth=0.5, saturation = 1,gap = 0.5,fliersize = 0, boxprops=dict(alpha=0.75))
        ax = plt.gca()
        xticks = ax.get_xticks()
        ax.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%H:%M:%S') for tm in xticks],rotation=75)
        ax.grid(True)  
        #plt.title(race_sel + ": LapAvg", fontsize=16)
        plt.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)
        #plt.show()
        
elif plot_sel_1 == "Multiple Venues":
    fig = plt.figure(figsize=(14, 10))
    if df.shape[0] > 0:
        sns.swarmplot(data = df, x = x_sel_1, y = 'Venue', hue = 'Cat', palette = 'bright', size = 4, dodge = True, legend = False)
        sns.boxplot(data = df, x = x_sel_1, y = 'Venue', hue = 'Cat', palette = 'dark', linewidth=0.5, saturation = 1,gap = 0.5,fliersize = 1, boxprops=dict(alpha=0.75))
        ax = plt.gca()
        xticks = ax.get_xticks()
        ax.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%H:%M:%S') for tm in xticks],rotation=75)
        ax.grid(True)  
        #plt.title(race_sel + ": LapAvg", fontsize=16)
        plt.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.show()
    
elif plot_sel_1 == "Series Ranking":
    fig = plt.figure(figsize=(14, 10))
    if df.shape[0] > 0:
        # sns.swarmplot(data = df, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, size = 4, dodge = False, legend = True)
        # df_tmp = df[(df['Upgrd_1'] == 'GRADUATED')]
        # sns.swarmplot(data = df_tmp, x = 'SeriesPoints', y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 'X', size = 5, dodge = False, legend = False)
        # df_tmp = df[(df['Upgrd_1'] == 'MANDATORY PERFORMANCE')]
        # sns.swarmplot(data = df_tmp, x = 'SeriesRank', y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = '^', size = 6, dodge = False, legend = False)        
        # df_tmp = df[(df['Upgrd_1'] == 'MAXED OUT')]
        # sns.swarmplot(data = df_tmp, x = 'SeriesRank', y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = '>', size = 6, dodge = False, legend = False)  
        # df_tmp = df[(df['Upgrd_1'] == 'GRADE BASED')]
        # sns.swarmplot(data = df_tmp, x = 'SeriesRank', y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 's', size = 4, dodge = False, legend = False)     
        # df_tmp = df[(df['Upgrd_1'] == 'NONE')]
        # sns.swarmplot(data = df_tmp, x = 'SeriesRank', y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, marker = 'o', size = 4, dodge = False, legend = True)
        sns.scatterplot(data = df, x = x_sel_1, y = 'Cat', hue = df['Upgrd_1'], palette = key_palette, style = df['Upgrd_1'], markers = marker_palette)
        
        
        #sns.boxplot(data = df, x = x_sel_1, y = 'Cat', hue = 'Laps_Course', palette = lap_palette, linewidth=0.5, saturation = 1,gap = 0.5,fliersize = 0, boxprops=dict(alpha=0.75))
        ax = plt.gca()
        if x_sel_1 == 'SeriesPoints':
            ax.invert_xaxis()
        #xticks = ax.get_xticks()
        #ax.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%H:%M:%S') for tm in xticks],rotation=75)
        ax.grid(True)  
        #plt.title(race_sel + ": LapAvg", fontsize=16)
        plt.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)
        #plt.show()

    
