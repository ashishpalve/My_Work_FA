#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('source activate python3')
#get_ipython().system('pip3 install pymongo')
#get_ipython().system('pip3 install dnspython')
#get_ipython().system('pip3 install pymongo[srv]')
#get_ipython().system('source deactivate')


# In[21]:


import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import configparser


# In[22]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# # MongoDB Connection

# In[23]:


config = configparser.ConfigParser()
config.read('reco_config_Tag_Classification.ini')
config.sections()


# In[24]:


mongodb_read_url = config['Connection']['mongodb_read_url']
mongodb_read_url


# In[25]:


from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(mongodb_read_url)


# In[26]:


db=client['prod-dump']


# In[ ]:





# # Feeds Data

# In[27]:


collection = db.feeds
feeds_df = pd.DataFrame(list(collection.find()))
print("Feeds_Data: ", feeds_df.shape)


# In[28]:


feeds_df = feeds_df.drop(columns = ['__v'])
feeds_df = feeds_df.rename(columns = {"_id":"contentId"})
feeds_df['contentId'] = [str(st) for st in feeds_df['contentId']]
feeds_df['authorId'] = [str(st) for st in feeds_df['authorId']]
feeds_df['resource_videoUrl'] = [st['videoUrl'] for st in feeds_df['resource']]
feeds_df['resource_image'] = [st['image'] for st in feeds_df['resource']]


# In[30]:


feeds_df['createdAt'] = pd.to_datetime(feeds_df['createdAt'])
feeds_df['dt'] = feeds_df['createdAt'].dt.date
feeds_df['mnth'] = feeds_df['createdAt'].dt.month
feeds_df['yr'] = feeds_df['createdAt'].dt.year
feeds_df['yr_mnth'] = feeds_df['yr'].map(str) + '-' + feeds_df['mnth'].map(str)


# In[32]:


#feeds_df_to_save = feeds_df[['contentId', 'anonymous', 'authorId', 'createdAt', 'isActive', 'isDelete', 'points', 'text', 'type', 'updatedAt', 'resource_videoUrl', 'resource_image']]
#feeds_df_to_save.to_csv(os.getcwd()+'/Datasets/feeds_df_for_reco.csv', index=False)


# ### Visualise Feed Data Pattern

# In[34]:


feeds_df_vis = feeds_df.copy()
feeds_by_dt = feeds_df_vis.groupby(['dt'])['contentId'].count().reset_index()


# In[35]:


#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(feeds_by_dt['dt'], feeds_by_dt['contentId'])
plt.xlabel("Date")
plt.ylabel("Number of Post")
plt.title("Number of Post per Day")

plt.savefig(os.getcwd()+'/Charts/Total_Post_Per_Day.png')

plt.show()
plt.close()


# In[36]:


feeds_by_mnth = feeds_df_vis.groupby(['yr_mnth'])['contentId'].count().reset_index()


# In[37]:


#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 8, 4
plt.style.use('ggplot')
plt.bar(feeds_by_mnth['yr_mnth'], feeds_by_mnth['contentId'])
plt.xlabel("Year-Month")
plt.ylabel("Number of Post")

plt.title("Number of Post each Month")

plt.savefig(os.getcwd()+'/Charts/Total_Post_each_Month.png')

plt.show()
plt.close()


# In[38]:


feeds_by_mnth_avg = feeds_df_vis.groupby(['yr_mnth'])['contentId'].count().reset_index()
feeds_by_mnth_avg['days_ct'] = feeds_df_vis.groupby(['yr_mnth'])['dt'].nunique().to_list()
feeds_by_mnth_avg['avg_post_per_day'] = feeds_by_mnth_avg['contentId']/feeds_by_mnth_avg['days_ct']
#print(feeds_by_mnth_avg.shape)


# In[39]:


#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 8, 4
plt.style.use('ggplot')
plt.bar(feeds_by_mnth_avg['yr_mnth'], feeds_by_mnth_avg['avg_post_per_day'])
plt.xlabel("Year-Month")
plt.ylabel("Average Post Per Day")

plt.title("Average Post per Day each Month")

plt.savefig(os.getcwd()+'/Charts/Avg_Post_Per_Day_each_Month.png')

plt.show()
plt.close()


# In[42]:


for_feed_tags = feeds_df[['contentId', 'authorId', 'createdAt', 'tagIds', 'text', 'type', 'dt', 'mnth', 'yr', 'yr_mnth']].copy()
#print(for_feed_tags.shape)


# ### Long format of Feeds data for Feed to Hashtag Mapping

# In[43]:


def wide_to_long_feeds_data(i):
    tags = [str(st) for st in for_feed_tags['tagIds'][i]]
    temp_df = pd.DataFrame(index = range(len(tags)), columns = for_feed_tags.columns)
    temp_df['contentId'] = for_feed_tags['contentId'][i]
    temp_df['authorId'] = for_feed_tags['authorId'][i]
    temp_df['createdAt'] = for_feed_tags['createdAt'][i]
    temp_df['tagIds'] = tags
    temp_df['text'] = for_feed_tags['text'][i]
    temp_df['type'] = for_feed_tags['type'][i]
    temp_df['dt'] = for_feed_tags['dt'][i]
    temp_df['mnth'] = for_feed_tags['mnth'][i]
    temp_df['yr'] = for_feed_tags['yr'][i]
    temp_df['yr_mnth'] = for_feed_tags['yr_mnth'][i]
    return(temp_df)


# In[44]:


results = [wide_to_long_feeds_data(x) for x in range(for_feed_tags.shape[0])]
feed2tags_df = pd.concat(results)
del results


# In[ ]:





# # Hashtags

# In[52]:


collection = db.hashtags
hashtags_df = pd.DataFrame(list(collection.find()))
hashtags_df = hashtags_df.drop(columns = ['__v'])
hashtags_df = hashtags_df.rename(columns = {"_id":"hashtagId"})
hashtags_df['hashtagId'] = [str(st) for st in hashtags_df['hashtagId']]
hashtags_df['authorId'] = [str(st) for st in hashtags_df['authorId']]
hashtags_df = hashtags_df.sort_values(by=['content'])
hashtags_df = hashtags_df[['hashtagId', 'authorId', 'content', 'isActive', 'isPrimary']]


# In[57]:


print("Unique hashtag ID:", len(hashtags_df['hashtagId'].unique()))
print("Unique hashtag:", len(hashtags_df['content'].unique()))
print("Unique authors:", len(hashtags_df['authorId'].unique()))
print("Primary Hashtags:", hashtags_df['isPrimary'].sum())


# In[58]:


hashtag_summary = hashtags_df.groupby(['content','isPrimary'])['hashtagId'].count().reset_index()
hashtag_summary = hashtag_summary.rename(columns = {"hashtagId": "hashtag_count"})
hashtag_summary = hashtag_summary.sort_values(by = ['hashtag_count'], ascending = False)
hashtag_summary.head(10)


# In[ ]:





# # Feed and Hashtag Data

# In[65]:


#print("feed & hashtag ID Data: ", feed2tags_df.shape)
#print("hashtag Data: ", hashtags_df.shape)
feed_hashtag = pd.merge(feed2tags_df, hashtags_df, left_on = ['tagIds'], right_on = ['hashtagId'], how = "left")
feed_hashtag = feed_hashtag.rename(columns = {'content':'hashtag', 'authorId_x':'authorId_content', 'authorId_y':'authorId_hashtag'})
#print("feed & hashtag Data after join: ", feed_hashtag.shape)
feed_hashtag = feed_hashtag.drop_duplicates()
feed_hashtag.head()


# In[67]:


#feed_hashtag.to_csv(os.getcwd()+'/Datasets/feed_hashtag_long.csv', index = False)


# In[68]:


print("Data Row Count:", feed_hashtag.shape[0])
print("unique feed ID:", len(feed_hashtag['contentId'].unique()))
print("unique feed:", len(feed_hashtag['text'].unique()))
print("unique feed author ID:", len(feed_hashtag['authorId_content'].unique()))
print("unique tag ID:", len(feed_hashtag['tagIds'].unique()))
print("unique hashtags:", len(feed_hashtag['hashtag'].unique()))
print("unique hashtag author ID:", len(feed_hashtag['authorId_hashtag'].unique()))


# In[ ]:





# In[70]:


feed_hashtag_summary = feed_hashtag.groupby(['hashtag', 'type', 'isPrimary'])['text'].agg([('unique_count','nunique'), ('total_count','count')]).reset_index()
feed_hashtag_summary = feed_hashtag_summary.sort_values(['unique_count'], ascending = False)
feed_hashtag_summary.head()


# In[71]:


feed_hashtag_summary_wide = feed_hashtag_summary.pivot_table(index = ['hashtag', 'isPrimary'], columns = 'type', values = ['unique_count', 'total_count']).reset_index()
feed_hashtag_summary_wide.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in feed_hashtag_summary_wide.columns]
feed_hashtag_summary_wide = feed_hashtag_summary_wide.fillna(0)
feed_hashtag_summary_wide['total_count'] = feed_hashtag_summary_wide['total_count_ARTICLE'] + feed_hashtag_summary_wide['total_count_POST'] + feed_hashtag_summary_wide['total_count_QUESTION']
feed_hashtag_summary_wide['unique_count'] = feed_hashtag_summary_wide['unique_count_ARTICLE'] + feed_hashtag_summary_wide['unique_count_POST'] + feed_hashtag_summary_wide['unique_count_QUESTION']
feed_hashtag_summary_wide = feed_hashtag_summary_wide[['hashtag', 'isPrimary', 'total_count', 'unique_count', 'total_count_ARTICLE', 'total_count_POST', 'total_count_QUESTION', 'unique_count_ARTICLE', 'unique_count_POST', 'unique_count_QUESTION']]
feed_hashtag_summary_wide = feed_hashtag_summary_wide.sort_values(['total_count'], ascending = False)
feed_hashtag_summary_wide.head(10)


# In[77]:


#feed_hashtag_summary_wide[feed_hashtag_summary_wide['isPrimary']==True]


# In[73]:


hashtag_per_feed = feed_hashtag.groupby(['contentId'])['hashtag'].count().reset_index()
hashtag_per_feed = hashtag_per_feed.rename(columns = {'hashtag':'hashtag_ct'})

no_of_tag_dist = hashtag_per_feed.groupby(['hashtag_ct'])['contentId'].count().reset_index()
no_of_tag_dist = no_of_tag_dist.rename(columns = {'contentId': 'contentId_ct'})
no_of_tag_dist.head()


# In[74]:


print("Average Hashtags in a Post: ", hashtag_per_feed['hashtag_ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(no_of_tag_dist['hashtag_ct'], no_of_tag_dist['contentId_ct'])
plt.xlabel("Number of Hashtag in a Post")
plt.ylabel("Number of Post")

plt.title("Number of Hashtag in a Post")

plt.savefig(os.getcwd()+'/Charts/No_of_Hashtag_in_Post.png')

plt.show()
plt.close()


# In[75]:


hashtag_per_feed_by_mnth = feed_hashtag.groupby(['yr_mnth', 'contentId'])['hashtag'].count().reset_index()
hashtag_per_feed_by_mnth = hashtag_per_feed_by_mnth.rename(columns = {'hashtag':'hashtag_ct'})
hashtag_per_feed_by_mnth.head()

avg_hashtag_by_mnth = hashtag_per_feed_by_mnth.groupby(['yr_mnth'])['hashtag_ct'].mean().reset_index()
avg_hashtag_by_mnth


# In[76]:


print("Average Hashtags in a Post: ", hashtag_per_feed['hashtag_ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(avg_hashtag_by_mnth['yr_mnth'], avg_hashtag_by_mnth['hashtag_ct'])
plt.xlabel("Year-Month")
plt.ylabel("Average Hashtag in a Post")

plt.title("Average Hashtag per post each Month")

plt.savefig(os.getcwd()+'/Charts/Avg_Hashtag_Per_Post_each_Month.png')


plt.show()
plt.close()


# In[ ]:




