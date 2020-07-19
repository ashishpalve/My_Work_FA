#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase 
from email import encoders 

mail_content = "Hello The Tag Classification model is refreshed at " + str(datetime.datetime.today()) + ". Below is the attached log."

#The mail addresses and password
sender_address = 'kk.recoengine@gmail.com'
sender_pass = 'recommendation@123'
receiver_address = 'ashishpalve.07@gmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'Tag Classification Refreshed at ' + str(datetime.datetime.today())   #The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))

# open the file to be sent  
filename = "logfile.txt"
attachment = open("/home/ubuntu/Tag_Classification_Solution/logfile.txt", "rb") 
  
# instance of MIMEBase and named as p 
p = MIMEBase('application', 'octet-stream') 
  
# To change the payload into encoded form 
p.set_payload((attachment).read()) 
  
# encode into base64 
encoders.encode_base64(p) 
   
p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  
# attach the instance 'p' to instance 'msg' 
message.attach(p) 

#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')

