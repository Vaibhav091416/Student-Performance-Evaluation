#to keep log of whatever is being executed, changed
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"          #creates and names the file
log_paths=os.path.join(os.getcwd(),'logs',LOG_FILE)                      #produces the path for the log file
os.makedirs(log_paths,exist_ok=True)                                     #since "logs" file don't exist already it creates it for first iteration and ensures that it's not reproduced again and again

LOG_FILE_PATH=os.path.join(log_paths,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(name)s - %(message)s",
    level=logging.INFO,
)

