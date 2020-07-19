#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('source activate python3')
#get_ipython().system('pip3 install pymongo')
#get_ipython().system('pip3 install dnspython')
#get_ipython().system('pip3 install pymongo[srv]')
#get_ipython().system('source deactivate')


# In[2]:


import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


# # MongoDB Connection

# In[3]:


from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb+srv://dev:oAX95s3vvOcqwZ4b@staging.wztxj.mongodb.net/test?retryWrites=true&w=majority")


# In[4]:


for db in client.list_databases():
    print(db)


# In[5]:


db=client['prod-dump']


# # Feeds Data

# In[6]:


collection = db.feeds
feeds_df = pd.DataFrame(list(collection.find()))
print(feeds_df.shape)
#feeds_df.head()


# In[7]:


feeds_df = feeds_df.drop(columns = ['__v'])
feeds_df = feeds_df.rename(columns = {"_id":"contentId"})


# In[8]:


feeds_df['contentId'] = [str(st) for st in feeds_df['contentId']]
feeds_df['authorId'] = [str(st) for st in feeds_df['authorId']]
print(feeds_df.shape)
#feeds_df.head()


# In[9]:


#feeds_df['resource_link'] = [st['link'] for st in feeds_df['resource']]
feeds_df['resource_videoUrl'] = [st['videoUrl'] for st in feeds_df['resource']]
feeds_df['resource_image'] = [st['image'] for st in feeds_df['resource']]


# In[10]:


feeds_df['createdAt'] = pd.to_datetime(feeds_df['createdAt'])
feeds_df['dt'] = feeds_df['createdAt'].dt.date
feeds_df['mnth'] = feeds_df['createdAt'].dt.month
feeds_df['yr'] = feeds_df['createdAt'].dt.year
feeds_df['yr_mnth'] = feeds_df['yr'].map(str) + '-' + feeds_df['mnth'].map(str)
#feeds_df.head()


# In[11]:


feeds_df_to_save = feeds_df[['contentId', 'anonymous', 'authorId', 'createdAt', 'isActive', 'isDelete', 'points', 'text', 'type', 'updatedAt', 'resource_videoUrl', 'resource_image']]
feeds_df_to_save.to_csv(os.getcwd()+'/Datasets/feeds_df_for_reco.csv', index=False)


# ### Visualise Feed Data Pattern

# In[12]:


feeds_df_vis = feeds_df.copy()


# In[13]:


feeds_by_dt = feeds_df_vis.groupby(['dt'])['contentId'].count().reset_index()
#print(feeds_by_dt.shape)
#feeds_by_dt.head()


# In[18]:


#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(feeds_by_dt['dt'], feeds_by_dt['contentId'])
plt.xlabel("Date")
plt.ylabel("Number of Post")
plt.title("Number of Post per Day")

plt.savefig(os.getcwd()+'/Charts/Total_Post_Per_Day.png')

plt.show()


# In[19]:


feeds_by_mnth = feeds_df_vis.groupby(['yr_mnth'])['contentId'].count().reset_index()
#print(feeds_by_mnth.shape)
#feeds_by_mnth.head()


# In[20]:


#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 8, 4
plt.style.use('ggplot')
plt.bar(feeds_by_mnth['yr_mnth'], feeds_by_mnth['contentId'])
plt.xlabel("Year-Month")
plt.ylabel("Number of Post")

plt.title("Number of Post each Month")

plt.savefig(os.getcwd()+'/Charts/Total_Post_each_Month.png')

plt.show()


# In[21]:


feeds_by_mnth_avg = feeds_df_vis.groupby(['yr_mnth'])['contentId'].count().reset_index()
feeds_by_mnth_avg['days_ct'] = feeds_df_vis.groupby(['yr_mnth'])['dt'].nunique().to_list()
feeds_by_mnth_avg['avg_post_per_day'] = feeds_by_mnth_avg['contentId']/feeds_by_mnth_avg['days_ct']
print(feeds_by_mnth_avg.shape)
#feeds_by_mnth_avg.head()


# In[22]:


#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 8, 4
plt.style.use('ggplot')
plt.bar(feeds_by_mnth_avg['yr_mnth'], feeds_by_mnth_avg['avg_post_per_day'])
plt.xlabel("Year-Month")
plt.ylabel("Average Post Per Day")

plt.title("Average Post per Day each Month")

plt.savefig(os.getcwd()+'/Charts/Avg_Post_Per_Day_each_Month.png')

plt.show()


# In[23]:


for_feed_tags = feeds_df[['contentId', 'authorId', 'createdAt', 'tagIds', 'text', 'type', 'dt', 'mnth', 'yr', 'yr_mnth']].copy()
print(for_feed_tags.shape)
for_feed_tags.head()


# ### Long format of Feeds data for Feed to Hashtag Mapping

# In[24]:


feed2tags_df = pd.DataFrame(columns = for_feed_tags.columns)
#feed2tags_df


# In[25]:


for i in range(for_feed_tags.shape[0]):
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
    
    feed2tags_df = feed2tags_df.append(temp_df)


# In[26]:


print(feed2tags_df.shape)
#feed2tags_df.head()


# # Hashtags

# In[27]:


collection = db.hashtags
hashtags_df = pd.DataFrame(list(collection.find()))
print(hashtags_df.shape)
#hashtags_df.head()


# In[28]:


hashtags_df = hashtags_df.drop(columns = ['__v'])
hashtags_df = hashtags_df.rename(columns = {"_id":"hashtagId"})


# In[29]:


hashtags_df['hashtagId'] = [str(st) for st in hashtags_df['hashtagId']]
hashtags_df['authorId'] = [str(st) for st in hashtags_df['authorId']]


# In[30]:


print(hashtags_df['content'].count())
print(hashtags_df['content'].nunique())
hashtags_df = hashtags_df.sort_values(by=['content'])
#hashtags_df.head()


# In[31]:


hashtags_df = hashtags_df[['hashtagId', 'authorId', 'content', 'isActive', 'isPrimary']]
print(hashtags_df.shape)
#hashtags_df.head()


# In[32]:


print("Unique hashtag ID:", len(hashtags_df['hashtagId'].unique()))
print("Unique hashtag:", len(hashtags_df['content'].unique()))
print("Unique authors:", len(hashtags_df['authorId'].unique()))
print("Primary Hashtags:", hashtags_df['isPrimary'].sum())


# In[33]:


hashtag_summary = hashtags_df.groupby(['content','isPrimary'])['hashtagId'].count().reset_index()
hashtag_summary = hashtag_summary.rename(columns = {"hashtagId": "hashtag_count"})
hashtag_summary = hashtag_summary.sort_values(by = ['hashtag_count'], ascending = False)
print(hashtag_summary.shape)
#hashtag_summary.head(10)


# In[ ]:





# # Feed and Hashtag Data

# In[34]:


print("feed & hashtag ID Data: ", feed2tags_df.shape)
print("hashtag Data: ", hashtags_df.shape)
feed_hashtag = pd.merge(feed2tags_df, hashtags_df, left_on = ['tagIds'], right_on = ['hashtagId'], how = "left")
feed_hashtag = feed_hashtag.rename(columns = {'content':'hashtag', 'authorId_x':'authorId_content', 'authorId_y':'authorId_hashtag'})
print("feed & hashtag Data after join: ", feed_hashtag.shape)
#feed_hashtag.head()


# In[35]:


print(feed_hashtag.shape)
feed_hashtag = feed_hashtag.drop_duplicates()
print(feed_hashtag.shape)


# In[36]:


feed_hashtag.to_csv(os.getcwd()+'/Datasets/feed_hashtag_long.csv', index = False)


# In[37]:


print("Data Row Count:", feed_hashtag.shape[0])
print("unique feed ID:", len(feed_hashtag['contentId'].unique()))
print("unique feed:", len(feed_hashtag['text'].unique()))
print("unique feed author ID:", len(feed_hashtag['authorId_content'].unique()))
print("unique tag ID:", len(feed_hashtag['tagIds'].unique()))
print("unique hashtags:", len(feed_hashtag['hashtag'].unique()))
print("unique hashtag author ID:", len(feed_hashtag['authorId_hashtag'].unique()))


# In[38]:


feed_hashtag_summary = feed_hashtag.groupby(['hashtag', 'type', 'isPrimary'])['text'].agg([('unique_count','nunique'), ('total_count','count')]).reset_index()
feed_hashtag_summary = feed_hashtag_summary.sort_values(['unique_count'], ascending = False)
print("Hashtags:", feed_hashtag_summary.shape[0])
#feed_hashtag_summary.head()


# In[39]:


feed_hashtag_summary_wide = feed_hashtag_summary.pivot_table(index = ['hashtag', 'isPrimary'], columns = 'type', values = ['unique_count', 'total_count']).reset_index()
feed_hashtag_summary_wide.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in feed_hashtag_summary_wide.columns]
feed_hashtag_summary_wide = feed_hashtag_summary_wide.fillna(0)
feed_hashtag_summary_wide['total_count'] = feed_hashtag_summary_wide['total_count_ARTICLE'] + feed_hashtag_summary_wide['total_count_POST'] + feed_hashtag_summary_wide['total_count_QUESTION']
feed_hashtag_summary_wide['unique_count'] = feed_hashtag_summary_wide['unique_count_ARTICLE'] + feed_hashtag_summary_wide['unique_count_POST'] + feed_hashtag_summary_wide['unique_count_QUESTION']
feed_hashtag_summary_wide = feed_hashtag_summary_wide[['hashtag', 'isPrimary', 'total_count', 'unique_count', 'total_count_ARTICLE', 'total_count_POST', 'total_count_QUESTION', 'unique_count_ARTICLE', 'unique_count_POST', 'unique_count_QUESTION']]
feed_hashtag_summary_wide = feed_hashtag_summary_wide.sort_values(['total_count'], ascending = False)
feed_hashtag_summary_wide.head(10)


# In[40]:


feed_hashtag_summary_wide[feed_hashtag_summary_wide['isPrimary']==True]


# In[41]:


hashtag_per_feed = feed_hashtag.groupby(['contentId'])['hashtag'].count().reset_index()
hashtag_per_feed = hashtag_per_feed.rename(columns = {'hashtag':'hashtag_ct'})

no_of_tag_dist = hashtag_per_feed.groupby(['hashtag_ct'])['contentId'].count().reset_index()
no_of_tag_dist = no_of_tag_dist.rename(columns = {'contentId': 'contentId_ct'})
#no_of_tag_dist.head()


# In[43]:


print("Average Hashtags in a Post: ", hashtag_per_feed['hashtag_ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(no_of_tag_dist['hashtag_ct'], no_of_tag_dist['contentId_ct'])
plt.xlabel("Number of Hashtag in a Post")
plt.ylabel("Number of Post")

plt.title("Number of Hashtags in Post")

plt.savefig(os.getcwd()+'/Charts/Hashtags_in_Post.png')

plt.show()


# In[44]:


hashtag_per_feed_by_mnth = feed_hashtag.groupby(['yr_mnth', 'contentId'])['hashtag'].count().reset_index()
hashtag_per_feed_by_mnth = hashtag_per_feed_by_mnth.rename(columns = {'hashtag':'hashtag_ct'})
hashtag_per_feed_by_mnth.head()

avg_hashtag_by_mnth = hashtag_per_feed_by_mnth.groupby(['yr_mnth'])['hashtag_ct'].mean().reset_index()
avg_hashtag_by_mnth


# In[45]:


print("Average Hashtags in a Post: ", hashtag_per_feed['hashtag_ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(avg_hashtag_by_mnth['yr_mnth'], avg_hashtag_by_mnth['hashtag_ct'])
plt.xlabel("Year-Month")
plt.ylabel("Average Hashtag in a Post")

plt.title("Average Hashtags in Post each Month")

plt.savefig(os.getcwd()+'/Charts/Avg_Hashtags_in_Post_each_Month.png')

plt.show()


# # Likes data from Likes Collection

# In[46]:


collection = db.likes
likes_coll_df = pd.DataFrame(list(collection.find()))
print(likes_coll_df.shape)
#likes_coll_df.head()


# In[47]:


likes_coll_df = likes_coll_df.drop(['__v'], axis=1)
likes_coll_df = likes_coll_df.rename(columns={'_id':'likeId'})
#likes_coll_df.head()


# In[48]:


likes_coll_df['likeId'] = [str(st) for st in likes_coll_df['likeId']]
likes_coll_df['commentId'] = [str(st) for st in likes_coll_df['commentId']]
likes_coll_df['referenceId'] = [str(st) for st in likes_coll_df['referenceId']]
likes_coll_df['userId'] = [str(st) for st in likes_coll_df['userId']]


# In[49]:


print("Total Like Count:", likes_coll_df['likeId'].count())
print("Total Content Like:", likes_coll_df[~(likes_coll_df['referenceId']=='nan')]['likeId'].count())
print("Unique Content Like:", likes_coll_df[~(likes_coll_df['referenceId']=='nan')]['referenceId'].nunique())
print("Total Comment Like:", likes_coll_df[~(likes_coll_df['commentId']=='nan')]['likeId'].count())
print("Unique Comment Like:", likes_coll_df[~(likes_coll_df['commentId']=='nan')]['commentId'].nunique())


# In[50]:


print(likes_coll_df.shape)
likes_coll_df = likes_coll_df.drop_duplicates()
print(likes_coll_df.shape)


# In[51]:


likes_coll_df.to_csv(os.getcwd()+'/Datasets/likes_coll_df.csv', index=False)


# In[ ]:





# In[52]:


content_like_summary = likes_coll_df[~(likes_coll_df['referenceId'] == 'nan')].groupby(['referenceId'])['likeId'].count().reset_index()
content_like_summary = content_like_summary.rename(columns = {'likeId':'likeId_Ct'})
content_like_summary = content_like_summary.sort_values(['likeId_Ct'], ascending = False)

no_of_like_dist = content_like_summary.groupby(['likeId_Ct'])['referenceId'].count().reset_index()
no_of_like_dist = no_of_like_dist.rename(columns = {'referenceId':'referenceId_Ct'})
#no_of_like_dist.head()


# In[53]:


print("Average Likes on a Post: ", content_like_summary['likeId_Ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(no_of_like_dist['likeId_Ct'], no_of_like_dist['referenceId_Ct'])
plt.xlabel("Number of Likes on a Post")
plt.ylabel("Number of Post")

plt.title("Number of Likes in Post")

plt.savefig(os.getcwd()+'/Charts/Likes_in_Post.png')

plt.show()


# In[54]:


likes_on_post_in_mnth = pd.merge(likes_coll_df[~(likes_coll_df['referenceId'] == 'nan')], feeds_df[['contentId', 'yr_mnth']], how = 'left', left_on = 'referenceId', right_on = 'contentId')

likes_per_feed_by_mnth = likes_on_post_in_mnth.groupby(['yr_mnth', 'referenceId'])['likeId'].count().reset_index()
likes_per_feed_by_mnth = likes_per_feed_by_mnth.rename(columns = {'likeId':'likeId_ct'})
likes_per_feed_by_mnth.head()

avg_likes_by_mnth = likes_per_feed_by_mnth.groupby(['yr_mnth'])['likeId_ct'].mean().reset_index()
avg_likes_by_mnth


# In[55]:


print("Average Likes on a Post: ", content_like_summary['likeId_Ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10, 4
plt.style.use('ggplot')
plt.bar(avg_likes_by_mnth['yr_mnth'], avg_likes_by_mnth['likeId_ct'])
plt.xlabel("Year-Month")
plt.ylabel("Average Likes on a Post")

plt.title("Avg Likes per post in each Month")

plt.savefig(os.getcwd()+'/Charts/Avg_Likes_per_Post_each_Month.png')

plt.show()


# # Bookmarks data from bookmarks collection

# In[56]:


collection = db.bookmarks
bookmarks_coll_df = pd.DataFrame(list(collection.find()))
print(bookmarks_coll_df.shape)
#bookmarks_coll_df.head()


# In[57]:


bookmarks_coll_df = bookmarks_coll_df.drop(['__v'], axis=1)
bookmarks_coll_df = bookmarks_coll_df.rename(columns = {'_id': 'bookmarkId'})
#bookmarks_coll_df.head()


# In[58]:


bookmarks_coll_df['bookmarkId'] = [str(st) for st in bookmarks_coll_df['bookmarkId']]
bookmarks_coll_df['referenceFeedId'] = [str(st) for st in bookmarks_coll_df['referenceFeedId']]
bookmarks_coll_df['userId'] = [str(st) for st in bookmarks_coll_df['userId']]
#bookmarks_coll_df.head()


# In[59]:


print(bookmarks_coll_df.shape)
bookmarks_coll_df = bookmarks_coll_df.drop_duplicates()
print(bookmarks_coll_df.shape)


# In[60]:


bookmarks_coll_df_non_comp = bookmarks_coll_df[~(bookmarks_coll_df['type']=='COMPANY')].copy()
print(bookmarks_coll_df.shape)
bookmarks_coll_df_non_comp.shape


# In[61]:


print('Total bookmarks:', bookmarks_coll_df_non_comp['bookmarkId'].nunique())
print('Unique Content with bookmark:', bookmarks_coll_df_non_comp['referenceFeedId'].nunique())
print('Unique Users with bookmark:', bookmarks_coll_df_non_comp['userId'].nunique())


# In[63]:


bookmarks_coll_df.to_csv(os.getcwd()+'/Datasets/bookmarks_coll_df.csv', index=False)


# # Comments Data

# In[64]:


collection = db.comments
comments_df_full = pd.DataFrame(list(collection.find()))
print(comments_df_full.shape)
#comments_df_full.head()


# In[65]:


comments_df_full = comments_df_full.drop(['__v'],axis=1)
comments_df_full = comments_df_full.rename(columns = {"_id":"commentId"})


# In[66]:


comments_df = comments_df_full[['commentId', 'authorId', 'createdAt', 'draftHtml', 'isActive', 'isSubComment', 'referenceId', 'type']].copy()


# In[67]:


#!pip3 install BeautifulSoup4


# In[68]:


from bs4 import BeautifulSoup


# In[69]:


def parser_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    text_list = soup.find_all('p')
    if(len(text_list) == 1):
        s = soup.find_all('p')[0].getText()
    else:
        s = ''
        for i in range(len(soup.find_all('p'))):
            s = s + soup.find_all('p')[i].getText()
    return(s)


# In[70]:


comments_df['comment_text'] = [parser_html(st) if len(st) > 0 else '' for st in comments_df['draftHtml']]


# In[71]:


comments_df = comments_df.drop(['draftHtml'], axis=1)


# In[72]:


comments_df['commentId'] = [str(st) for st in comments_df['commentId']]
comments_df['authorId'] = [str(st) for st in comments_df['authorId']]
comments_df['referenceId'] = [str(st) for st in comments_df['referenceId']]


# In[73]:


print(comments_df.shape)
comments_df = comments_df.drop_duplicates()
print(comments_df.shape)


# In[74]:


print("Data Size:", comments_df.shape[0])
print("Total Comments:", comments_df['commentId'].nunique())
print("Total Post with comments:", comments_df['referenceId'].nunique())
print("Total Primary Comments:", comments_df[comments_df['isSubComment'] == False]['commentId'].nunique())
print("Total SubComments:", comments_df[comments_df['isSubComment'] == True]['commentId'].nunique())


# In[75]:


comments_df.to_csv(os.getcwd()+'/Datasets/comments_df.csv', index = False)


# In[76]:


comment_summary = comments_df[~(comments_df['referenceId'] == 'nan')].groupby(['referenceId'])['commentId'].count().reset_index()
comment_summary = comment_summary.rename(columns = {'commentId':'commentId_Ct'})
comment_summary = comment_summary.sort_values(['commentId_Ct'], ascending = False)
#comment_summary.head()

no_of_comment_dist = comment_summary.groupby(['commentId_Ct'])['referenceId'].count().reset_index()
no_of_comment_dist = no_of_comment_dist.rename(columns = {'referenceId':'referenceId_Ct'})

#no_of_comment_dist.head()


# In[77]:


print("Average Comments on a Post: ", comment_summary['commentId_Ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 18, 8
plt.style.use('ggplot')
plt.bar(no_of_comment_dist['commentId_Ct'], no_of_comment_dist['referenceId_Ct'])
plt.xlabel("Number of Comments / Subcomments on a Post")
plt.ylabel("Number of Post")

plt.title("Number of Comments in a post")

plt.savefig(os.getcwd()+'/Charts/No_of_Comments_on_Post.png')

plt.show()


# In[78]:


comments_on_post_in_mnth = pd.merge(comments_df[~(comments_df['referenceId'] == 'nan')], feeds_df[['contentId', 'yr_mnth']], how = 'left', left_on = 'referenceId', right_on = 'contentId')

comments_per_feed_by_mnth = comments_on_post_in_mnth.groupby(['yr_mnth', 'referenceId'])['commentId'].count().reset_index()
comments_per_feed_by_mnth = comments_per_feed_by_mnth.rename(columns = {'commentId':'commentId_ct'})
comments_per_feed_by_mnth.head()

avg_comments_by_mnth = comments_per_feed_by_mnth.groupby(['yr_mnth'])['commentId_ct'].mean().reset_index()
avg_comments_by_mnth


# In[79]:


print("Average Comments on a Post: ", comment_summary['commentId_Ct'].mean())
#get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10, 4
plt.style.use('ggplot')
plt.bar(avg_comments_by_mnth['yr_mnth'], avg_comments_by_mnth['commentId_ct'])
plt.xlabel("Year-Month")
plt.ylabel("Average Comments on a Post")

plt.title("Average Comments on Post in each Month")

plt.savefig(os.getcwd()+'/Charts/Avg_Comments_per_Post_in_each_Month.png')

plt.show()


# In[ ]:




