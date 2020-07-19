#!/bin/bash     
cd /home/ubuntu/start_ec2/

logfile=/home/ubuntu/start_ec2/logfile.txt

exec &> $logfile

echo "Code Run Starts at" >> $logfile
echo $(date -u) >> $logfile
echo "-------------------------------" >> $logfile

aws ec2 start-instances --instance-ids i-02694c87f236327e6

echo "-------------------------------" >> $logfile
echo "Code Run Ends at" >> $logfile
echo $(date -u) >> $logfile
