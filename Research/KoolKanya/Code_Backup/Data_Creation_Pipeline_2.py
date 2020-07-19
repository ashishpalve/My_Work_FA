#!/usr/bin/env python
# coding: utf-8

# In[5]:


#get_ipython().system('source activate python3')
#get_ipython().system('pip3 install pymongo')
#get_ipython().system('pip3 install dnspython')
#get_ipython().system('pip3 install pymongo[srv]')
#get_ipython().system('source deactivate')


# In[6]:


import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from bs4 import BeautifulSoup
import configparser


# # MongoDB Connection

# In[7]:


print("-----------------------------------------------------------------------------------------------------")
print("-----------------------------  Data Creation  -------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------")


# In[10]:


config = configparser.ConfigParser()
config.read('reco_config.ini')
config.sections()


# In[11]:


mongodb_url = config['Connection']['mongodb_url']
mongodb_url


# In[12]:


print("-----------------------------------------------------------------------------------------------------")
print("\nConnecting to the database:")
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(mongodb_url)


# In[13]:


#for db in client.list_databases():
#    print(db)


# In[14]:


db=client['prod-dump']


# # Feeds Data

# In[15]:


print("-----------------------------------------------------------------------------------------------------")
print("\nReading Feeds Data")


# In[16]:


collection = db.feeds
feeds_df = pd.DataFrame(list(collection.find()))
#print(feeds_df.shape)
#feeds_df.head()


# In[17]:


feeds_df = feeds_df.drop(columns = ['__v'])
feeds_df = feeds_df.rename(columns = {"_id":"contentId"})


# In[18]:


feeds_df['contentId'] = [str(st) for st in feeds_df['contentId']]
feeds_df['authorId'] = [str(st) for st in feeds_df['authorId']]
#print(feeds_df.shape)
#feeds_df.head()


# In[19]:


#feeds_df['resource_link'] = [st['link'] for st in feeds_df['resource']]
feeds_df['resource_videoUrl'] = [st['videoUrl'] for st in feeds_df['resource']]
feeds_df['resource_image'] = [st['image'] for st in feeds_df['resource']]


# In[20]:


feeds_df['createdAt'] = pd.to_datetime(feeds_df['createdAt'])
feeds_df['dt'] = feeds_df['createdAt'].dt.date
feeds_df['mnth'] = feeds_df['createdAt'].dt.month
feeds_df['yr'] = feeds_df['createdAt'].dt.year
feeds_df['yr_mnth'] = feeds_df['yr'].map(str) + '-' + feeds_df['mnth'].map(str)
print("Feeds Data:", feeds_df.shape)
#feeds_df.head()


# In[ ]:





# In[13]:


#non_articles_df = feeds_df[feeds_df['type'] != 'ARTICLE'].copy()
#articles_df = feeds_df[feeds_df['type'] == 'ARTICLE'].copy()


# In[14]:


#def parser_article(text):
#    soup = BeautifulSoup(text, 'html.parser')
#    text_list = soup.find_all('p')
#    s = soup.find_all('p')

#    s = [st.getText() for st in soup.find_all('p')]
#    s2 = ''.join(s)
#    return(s2)


# In[17]:


#articles_df['parsed_text'] = [parser_article(st) if len(st) > 0 else '' for st in articles_df['content']]
#articles_df['article_text'] = articles_df['text'] + ' ' + articles_df['parsed_text']


# In[18]:


#articles_df['text'] = articles_df['article_text']


# In[19]:


#articles_df = articles_df.drop(['article_text'], axis=1)
#articles_df = articles_df.drop(['parsed_text'], axis=1)


# In[23]:


#feeds_df = non_articles_df.append(articles_df)


# In[ ]:





# In[24]:


#feeds_df_to_save = feeds_df[['contentId', 'anonymous', 'authorId', 'createdAt', 'isActive', 'isDelete', 'points', 'text', 'type', 'updatedAt', 'resource_videoUrl', 'resource_image']]
#feeds_df_to_save.to_csv(os.getcwd()+'/Datasets/feeds_df_for_reco.csv', index=False)


# ### Visualise Feed Data Pattern

# In[25]:


feeds_df_vis = feeds_df.copy()


# In[26]:


feeds_by_dt = feeds_df_vis.groupby(['dt'])['contentId'].count().reset_index()
#print(feeds_by_dt.shape)
#feeds_by_dt.head()


# In[27]:


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


# In[28]:


feeds_by_mnth = feeds_df_vis.groupby(['yr_mnth'])['contentId'].count().reset_index()
#print(feeds_by_mnth.shape)
#feeds_by_mnth.head()


# In[29]:


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


# In[30]:


feeds_by_mnth_avg = feeds_df_vis.groupby(['yr_mnth'])['contentId'].count().reset_index()
feeds_by_mnth_avg['days_ct'] = feeds_df_vis.groupby(['yr_mnth'])['dt'].nunique().to_list()
feeds_by_mnth_avg['avg_post_per_day'] = feeds_by_mnth_avg['contentId']/feeds_by_mnth_avg['days_ct']
#print(feeds_by_mnth_avg.shape)
#feeds_by_mnth_avg.head()


# In[31]:


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


# In[32]:


for_feed_tags = feeds_df[['contentId', 'authorId', 'createdAt', 'tagIds', 'text', 'type', 'dt', 'mnth', 'yr', 'yr_mnth']].copy()
#print(for_feed_tags.shape)
#for_feed_tags.head()


# # Likes data from Likes Collection

# In[33]:


print("-----------------------------------------------------------------------------------------------------")
print("\n Procesing Likes Data:")
collection = db.likes
likes_coll_df = pd.DataFrame(list(collection.find()))
#print(likes_coll_df.shape)
#likes_coll_df.head()


# In[34]:


likes_coll_df = likes_coll_df.drop(['__v'], axis=1)
likes_coll_df = likes_coll_df.rename(columns={'_id':'likeId'})
#likes_coll_df.head()


# In[35]:


likes_coll_df['likeId'] = [str(st) for st in likes_coll_df['likeId']]
likes_coll_df['commentId'] = [str(st) for st in likes_coll_df['commentId']]
likes_coll_df['referenceId'] = [str(st) for st in likes_coll_df['referenceId']]
likes_coll_df['userId'] = [str(st) for st in likes_coll_df['userId']]


# In[36]:


print("\nLikes Summary:")
print("Total Like Count:", likes_coll_df['likeId'].count())
print("Total Content Like:", likes_coll_df[~(likes_coll_df['referenceId']=='nan')]['likeId'].count())
print("Unique Content Like:", likes_coll_df[~(likes_coll_df['referenceId']=='nan')]['referenceId'].nunique())
print("Total Comment Like:", likes_coll_df[~(likes_coll_df['commentId']=='nan')]['likeId'].count())
print("Unique Comment Like:", likes_coll_df[~(likes_coll_df['commentId']=='nan')]['commentId'].nunique())


# In[37]:


#print(likes_coll_df.shape)
likes_coll_df = likes_coll_df.drop_duplicates()
#print(likes_coll_df.shape)


# In[38]:


#likes_coll_df.to_csv(os.getcwd()+'/Datasets/likes_coll_df.csv', index=False)


# In[ ]:





# In[39]:


content_like_summary = likes_coll_df[~(likes_coll_df['referenceId'] == 'nan')].groupby(['referenceId'])['likeId'].count().reset_index()
content_like_summary = content_like_summary.rename(columns = {'likeId':'likeId_Ct'})
content_like_summary = content_like_summary.sort_values(['likeId_Ct'], ascending = False)

no_of_like_dist = content_like_summary.groupby(['likeId_Ct'])['referenceId'].count().reset_index()
no_of_like_dist = no_of_like_dist.rename(columns = {'referenceId':'referenceId_Ct'})
#no_of_like_dist.head()


# In[40]:


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
plt.close()


# In[41]:


likes_on_post_in_mnth = pd.merge(likes_coll_df[~(likes_coll_df['referenceId'] == 'nan')], feeds_df[['contentId', 'yr_mnth']], how = 'left', left_on = 'referenceId', right_on = 'contentId')

likes_per_feed_by_mnth = likes_on_post_in_mnth.groupby(['yr_mnth', 'referenceId'])['likeId'].count().reset_index()
likes_per_feed_by_mnth = likes_per_feed_by_mnth.rename(columns = {'likeId':'likeId_ct'})
#likes_per_feed_by_mnth.head()

avg_likes_by_mnth = likes_per_feed_by_mnth.groupby(['yr_mnth'])['likeId_ct'].mean().reset_index()
#avg_likes_by_mnth


# In[42]:


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
plt.close()


# # Bookmarks data from bookmarks collection

# In[43]:


print("-----------------------------------------------------------------------------------------------------")
print("\nProcesing Bookmarks Data")
collection = db.bookmarks
bookmarks_coll_df = pd.DataFrame(list(collection.find()))
#print(bookmarks_coll_df.shape)
#bookmarks_coll_df.head()


# In[44]:


bookmarks_coll_df = bookmarks_coll_df.drop(['__v'], axis=1)
bookmarks_coll_df = bookmarks_coll_df.rename(columns = {'_id': 'bookmarkId'})
#bookmarks_coll_df.head()


# In[45]:


bookmarks_coll_df['bookmarkId'] = [str(st) for st in bookmarks_coll_df['bookmarkId']]
bookmarks_coll_df['referenceFeedId'] = [str(st) for st in bookmarks_coll_df['referenceFeedId']]
bookmarks_coll_df['userId'] = [str(st) for st in bookmarks_coll_df['userId']]
#bookmarks_coll_df.head()


# In[46]:


#print(bookmarks_coll_df.shape)
bookmarks_coll_df = bookmarks_coll_df.drop_duplicates()
#print(bookmarks_coll_df.shape)


# In[47]:


bookmarks_coll_df_non_comp = bookmarks_coll_df[~(bookmarks_coll_df['type']=='COMPANY')].copy()
#print(bookmarks_coll_df.shape)
#bookmarks_coll_df_non_comp.shape


# In[48]:


print("\nBookmarks Summary:")
print('Total bookmarks:', bookmarks_coll_df_non_comp['bookmarkId'].nunique())
print('Unique Content with bookmark:', bookmarks_coll_df_non_comp['referenceFeedId'].nunique())
print('Unique Users with bookmark:', bookmarks_coll_df_non_comp['userId'].nunique())


# In[49]:


#bookmarks_coll_df.to_csv(os.getcwd()+'/Datasets/bookmarks_coll_df.csv', index=False)


# # Comments Data

# In[50]:


print("-----------------------------------------------------------------------------------------------------")
print("\nProcesing Comments Data")
collection = db.comments
comments_df_full = pd.DataFrame(list(collection.find()))
#print(comments_df_full.shape)
#comments_df_full.head()


# In[51]:


comments_df_full = comments_df_full.drop(['__v'],axis=1)
comments_df_full = comments_df_full.rename(columns = {"_id":"commentId"})


# In[52]:


comments_df = comments_df_full[['commentId', 'authorId', 'createdAt', 'draftHtml', 'isActive', 'isSubComment', 'referenceId', 'type']].copy()


# In[53]:


#!pip3 install BeautifulSoup4


# In[54]:


from bs4 import BeautifulSoup


# In[55]:


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


# In[56]:


comments_df['comment_text'] = [parser_html(st) if len(st) > 0 else '' for st in comments_df['draftHtml']]


# In[57]:


comments_df = comments_df.drop(['draftHtml'], axis=1)


# In[58]:


comments_df['commentId'] = [str(st) for st in comments_df['commentId']]
comments_df['authorId'] = [str(st) for st in comments_df['authorId']]
comments_df['referenceId'] = [str(st) for st in comments_df['referenceId']]


# In[59]:


#print(comments_df.shape)
comments_df = comments_df.drop_duplicates()
#print(comments_df.shape)


# In[60]:


print("\nComments Summary:")
print("Data Size:", comments_df.shape[0])
print("Total Comments:", comments_df['commentId'].nunique())
print("Total Post with comments:", comments_df['referenceId'].nunique())
print("Total Primary Comments:", comments_df[comments_df['isSubComment'] == False]['commentId'].nunique())
print("Total SubComments:", comments_df[comments_df['isSubComment'] == True]['commentId'].nunique())


# In[61]:


#comments_df.to_csv(os.getcwd()+'/Datasets/comments_df.csv', index = False)


# In[62]:


comment_summary = comments_df[~(comments_df['referenceId'] == 'nan')].groupby(['referenceId'])['commentId'].count().reset_index()
comment_summary = comment_summary.rename(columns = {'commentId':'commentId_Ct'})
comment_summary = comment_summary.sort_values(['commentId_Ct'], ascending = False)
#comment_summary.head()

no_of_comment_dist = comment_summary.groupby(['commentId_Ct'])['referenceId'].count().reset_index()
no_of_comment_dist = no_of_comment_dist.rename(columns = {'referenceId':'referenceId_Ct'})

#no_of_comment_dist.head()


# In[63]:


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
plt.close()


# In[64]:


comments_on_post_in_mnth = pd.merge(comments_df[~(comments_df['referenceId'] == 'nan')], feeds_df[['contentId', 'yr_mnth']], how = 'left', left_on = 'referenceId', right_on = 'contentId')

comments_per_feed_by_mnth = comments_on_post_in_mnth.groupby(['yr_mnth', 'referenceId'])['commentId'].count().reset_index()
comments_per_feed_by_mnth = comments_per_feed_by_mnth.rename(columns = {'commentId':'commentId_ct'})
#comments_per_feed_by_mnth.head()

avg_comments_by_mnth = comments_per_feed_by_mnth.groupby(['yr_mnth'])['commentId_ct'].mean().reset_index()
avg_comments_by_mnth


# In[65]:


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
plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




