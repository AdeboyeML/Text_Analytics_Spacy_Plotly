import warnings
warnings.filterwarnings("ignore")

import cufflinks as cf
from collections import Counter
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
'exec(%matplotlib inline)'

# plotly
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.express as px


class plot_results:
    
    def __init__(self, novelname):
        
        self.novel = novelname
        
    """
    This class plots most of the text analyses results done by other classes using 
    Plotly and Seaborn
    """    
    #Plot Number of times specific entities where mentioned in the Novel
    def plot_entitycount(self, entity_ct, Entity, color_continuous_scale):
        
        if type(entity_ct) == dict:
            df_ent = pd.DataFrame(entity_ct.items(), columns = [Entity, 'counts'])
        else:
            df_ent = pd.DataFrame(entity_ct, columns = [Entity, 'counts'])
        
        df_ent = df_ent.sort_values(by='counts')
        fig = px.bar(df_ent, x= 'counts', y= Entity, orientation = 'h', 
                     hover_data=df_ent.columns, color='counts',
                     labels={'counts':'<b> Number of times mentioned <b>'}, width=1000, height=700,
                     color_continuous_scale=color_continuous_scale)
        
        fig.update_layout(title='<b> ' + Entity + ' mentioned in the ' + self.novel + ' <b>', xaxis_title='<b> Number of times mentioned <b>')
        
        iplot(fig)
        
    #Plot the Numbers of Chapters characters appear in    
    def plot_ctper(self, entity_ct, Entity, color_continuous_scale):
        
        df_ent = pd.DataFrame(entity_ct.items(), columns = [Entity, 'counts'])
        df_ent = df_ent.sort_values(by='counts')
        
        fig = px.bar(df_ent, x= 'counts', y= Entity, orientation = 'h', 
                     hover_data=df_ent.columns, color='counts',
                     labels={'counts':'<b> Number of times mentioned <b>'}, width=1000, height=700,
                     color_continuous_scale=color_continuous_scale)
        
        fig.update_layout(title='<b> Number of Chapters each ' + Entity + ' appeared in <b>', xaxis_title='<b> Number of Chapters <b>')
        iplot(fig)
        
    #Plot Gender distribution   
    def gender_plot(self, df):
        
        fig = px.pie(df, values='size', names='gender', 
                     title='Gender Distribution in ' + self.novel,
                     hover_data=df.columns)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.show()
        
    #Extract most common verbs and plot for all the Major characters 
    def most_common(self, df, num_of_verbs):
        
        most_common = (df.groupby(['Subject', 'Action_verb']).size().groupby(level=0, group_keys=False).nlargest(num_of_verbs).rename('Count')
                       .reset_index(level=1).rename(columns={'Action_verb': 'Most Common'}))
        
        fig = px.bar(most_common, x= 'Most Common', y= 'Count',
                     hover_data=most_common.columns, color=most_common.index,labels={'Most Common':'<b> Most Common Verbs mentioned <b>'}, width=1000, height=800, 
             color_continuous_scale=px.colors.sequential.Cividis)
        
        fig.update_layout(title='<b> Top ' + str(num_of_verbs) + ' action verbs used by each of the characters in the ' + self.novel + ' <b>', 
                          xaxis_title='<b> Most Common Verbs <b>', yaxis_title='<b> Number of times mentioned <b>', xaxis=dict(tickfont=dict(size=13)),
                          yaxis=dict(tickfont=dict(size=13)), legend_title_text='<b> Characters <b>',
                          legend=dict(x=0.89, y=1, traceorder="normal", font=dict(family="sans-serif", size=13, color="black"), 
                                      bgcolor="LightSteelBlue", borderwidth=2))
        iplot(fig)
        return most_common
    
    #Plot sentences in the novel where the characters are the "Subjects"
    def action_plot(self, df):
        sns.set(style='darkgrid')
        fig, ax = plt.subplots(figsize=(8,12), dpi=124*2)
        sns.stripplot(x='Sentence_Number', y='Subject', data=df, ax=ax, color='xkcd:cerulean',
                      jitter=0.25,dodge=True, palette=sns.husl_palette(2, l=0.5, s=.95),
                      size=5, linewidth=2)
        ax.set_title("Sentences in the " + self.novel + " where characters took actions, sorted by their first appearance")
    
    #Plot most common/top verbs used by specific characters
    def xter_verbs_plot(self, df, name, num_of_verb, colorscale):
        
        xter_verb = df[df['Subject'] == name]['Action_verb'].value_counts().head(num_of_verb)
        xter_verb = pd.DataFrame({'Verbs':xter_verb.index, 'Count':xter_verb.values}).sort_values(by = 'Count')
        
        fig = px.bar(xter_verb, x= 'Count', y= 'Verbs', orientation = 'h',
                     hover_data=xter_verb.columns, color='Count',
                     labels={'Count':'<b> Most Common Verbs mentioned <b>'}, width=1000, height=800,
                     color_continuous_scale=colorscale)
        
        fig.update_layout(title='<b> ' + str(num_of_verb) + ' most used verbs by ' + name + ' in the Novel, when ' + name + ' is the Subject of a sentence <b>', xaxis_title='<b> Most Common Verbs <b>',
                          yaxis_title='<b> Number of times mentioned <b>', xaxis=dict(tickfont=dict(size=13)),
                          yaxis=dict(tickfont=dict(size=13)), legend_title_text='<b> Characters <b>',
                          legend=dict(x=0.89, y=1, traceorder="normal", font=dict(family="sans-serif", size=13, color="black"), bgcolor="LightSteelBlue", borderwidth=2))
        iplot(fig)
        
    #Check the top 3 character's interaction with each other
    def top3xters_interaction(self, characters):
        top_3 = characters[:3]
        for i,x in enumerate(top_3):
            top1_2 = top_3[:2]
            top1_3 = top_3[::2]
            top2_3 = top_3[1:3]
            if i == 0:
                break
        return top1_2, top2_3, top1_3
    
    #Extract sentences where specific characters appear in and plot their appearances accross the entire novel
    def extract_sentences_plot(self, df, max_xtrs, color):
        
        max_cts = []
        xters_ct = []
        xter_sentences = []
        if type(max_xtrs) == list:
            for x in range(0, len(df), 1):
                kk = re.compile("({})+".format("|".join(re.escape(c) for c in max_xtrs)))
                xters = kk.findall(df.Contents[x])
                #print(type(xters))
                if len(xters) >= 1:
                    xter_sentences.append(df.Contents[x])
                else:
                    xter_sentences.append(np.NaN)
                xters_ct.append(xters)
        
        else:
            for x in range(0, len(df), 1):
                xters = re.findall(max_xtrs, df.Contents[x])
                if len(xters) >= 1:
                    xter_sentences.append(df.Contents[x])
                else:
                    xter_sentences.append(np.NaN)
                xters_ct.append(xters)
                    
        for x in range(0,len(xters_ct),1):
            xtrs = dict(Counter(xters_ct[x]).most_common())
            max_cts.append(xtrs)
            
        df_xter_sent = pd.DataFrame(max_cts)
        df_xter_sent['Contents'] = xter_sentences
        df_xter_sent.dropna(inplace = True)
        print(df_xter_sent.head())
        
        #drop sentence
        df_xter = df_xter_sent.drop(['Contents'], axis=1)
        #convert column names into strings
        content = ""
        
        if df_xter.columns.tolist() == ['I']:
            content = 'Appearances of the Unamed Narrator (represented as "I") in'
            
        elif len(df_xter.columns) == 1:
            content = ''.join(df_xter.columns)
            content = 'Appearances of ' + content + ' in'
        else:
            content = ' and '.join(df_xter.columns)
            content = 'Appearances of ' + content + ' when mentioned together in the same sentence in'
            
        fig = df_xter.iplot(asFigure = True, kind = 'bar', colors = color)
        fig.update_layout(title='<b> ' + content + ' ' + self.novel + ' <b>', 
                          xaxis_title='<b> Number of Sentences <b>', yaxis_title = '<b> Amount of Mentions <b>', width = 1000)
        
        fig.update_xaxes(dtick=200)
        iplot(fig)
        return df_xter_sent
