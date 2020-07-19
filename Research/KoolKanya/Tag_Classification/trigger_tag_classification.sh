#!/bin/bash     
cd /home/ubuntu/Tag_Classification_Solution/

logfile=/home/ubuntu/Tag_Classification_Solution/logfile.txt

exec &> $logfile

echo "Tag Classification Code Run Starts at" >> $logfile
echo $(date -u) >> $logfile
echo "-------------------------------" >> $logfile

python3 Tag_Classification_Modelling_Pipeline.py >> $logfile

echo "Tag Classification Code Run Ends at" >> $logfile
echo $(date -u) >> $logfile
echo "-------------------------------" >> $logfile

python3 Send_Mail.py

aws ec2 stop-instances --instance-ids i-02694c87f236327e6