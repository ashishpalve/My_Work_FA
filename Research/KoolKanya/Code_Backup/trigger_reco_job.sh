#!/bin/bash     
cd /home/ubuntu/reco/Reco_Solution/

logfile=/home/ubuntu/reco/Reco_Solution/logfile.txt

exec &> $logfile

echo "All Recommendation Code Run Starts at" >> $logfile
echo $(date -u) >> $logfile
echo "-------------------------------" >> $logfile

python3 Recommendation_Scoring_Engine_Pipeline.py >> $logfile

echo "All Recommendation Code Run Ends at" >> $logfile
echo $(date -u) >> $logfile
echo "-------------------------------" >> $logfile

echo "Article Recommendation Code Run Starts at" >> $logfile
echo $(date -u) >> $logfile
echo "-------------------------------" >> $logfile

python3 Item_to_Item_Recommendation.py >> $logfile

echo "-------------------------------" >> $logfile
echo "Article Recommendation Code Run Ends at" >> $logfile
echo $(date -u) >> $logfile

python3 Send_Mail.py

aws ec2 stop-instances --instance-ids i-02694c87f236327e6