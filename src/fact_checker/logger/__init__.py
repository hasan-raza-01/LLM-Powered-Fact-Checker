import os
import logging
import datetime

filename_format = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
filename = f"{filename_format}.log"

#####################################################
os.environ["LOGS_FOLDER"]="logs"    #################
#####################################################

folder_name = os.path.join(os.getcwd(), os.getenv("LOGS_FOLDER"))

os.makedirs(folder_name, exist_ok=True)

filepath = os.path.join(folder_name, filename)

logging.basicConfig(
    level=logging.INFO,
    filename=filepath,
    filemode="a",
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(pathname)s - %(lineno)s - %(message)s",
    datefmt="%d/%m/%Y-%H:%M:%S"
)


if __name__=="__main__":
    logging.info("logging has started")
    
