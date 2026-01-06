# humanfriendly cloudpathlib 

# from DpyUtils import ...
from humanfriendly import format_timespan
import os
from glob import glob
import time
from cloudpathlib import CloudPath
from cloudpathlib import S3Client

"""
**TODO:
    - Add a Secrets File for all Enviornment based Credentials in a file
    - 
"""  

def download_from_s3(s3_uri, download_dir, aws_access_key_id, aws_secret_access_key ):    
    start_time = time.time()
    client = S3Client(aws_access_key_id, aws_secret_access_key)
    cp = CloudPath(s3_uri, client)
    cp.download_to(download_dir)
    time_taken = time.time()-start_time
    files_count = len(glob('Solution2/**/*', recursive = True))
    print("Files Count : ", files_count)
    print("Time Taken : ", format_timespan(time_taken))