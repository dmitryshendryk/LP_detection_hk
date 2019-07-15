import argparse

import sys
import os
import glob
import logging.config

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from detect import *
from api.config import ConfigGlobalAPI, ConfigSuzhouAPI, ConfigHKAPI
from api.ftp_server import *
from api.mqtt_client import *
import time 
import shutil


REMOTE_FOLDER_PATH='/klubms/resources/license'

lp_path=os.path.join(ROOT_DIR,"weights/this_is_the_best_lp_weight_1203.h5") # "/home/ccs1/Documents/Projects/LP_detect/weights/this_is_the_best_lp_weight_1203.h5" #535
char_path =os.path.join(ROOT_DIR,"weights/mask_rcnn_plc_0535.h5")# "/home/ccs1/Documents/Projects/LP_detect_dep/weights/mask_rcnn_plc_0535.h5"
lp_model,char_model=load_model(lp_path,char_path)


def setup_logging(
    default_path=os.path.join(ROOT_DIR,'api/logging.json'),
    default_level=logging.INFO,
    env_key='LOG_CFG'):

    path = default_path
    value = os.getenv(env_key, None)

    print(path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)



def restart():
    python = sys.executable
    os.execl(python, python, * sys.argv)

def onerror(func, path, exc_info):
   
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def cron_job(mqtt_client, logger, sftp):
    print("Started SFT client")
    while True:

        # folder_path = os.listdir(os.path.join(ROOT_DIR,'imgs'))
        folder_path = os.listdir('/sftp/lpr/upload')
        folder_path = list(filter(lambda x: len(x.split('.')) == 1, folder_path))
        if folder_path:
            for folder in folder_path:
                imgs_path = os.path.join("/sftp/lpr/upload", folder)
                print("PROCESS FOLDER ", imgs_path)
                item=process(lp_model,char_model,imgs_path, logger)
                try:
                    # shutil.rmtree(imgs_path, onerror=onerror)
                    sftp.sftp_remove(imgs_path, os.listdir(imgs_path))
                    print("DELETED FOLDER ", imgs_path)
                except OSError as e:
                    print("Error: %s - %s." %(e.filename, e.strerror))
                try:

                    temp = {
                        "code": item['code'],
                        "message": item['message'],
                        "licencePlate": item['licencePlate'],
                        "vehicleType": '',
                        "keyImage": item['img_name'],
                        "vehicleColor":"color"}
                except KeyError:
                    continue
                try:
                    print("v1/recognition/vehicle/res/" + item['img_name'].split('.')[0])
                    mqtt_client.publish("v1/recognition/vehicle/res/" + item['img_name'].split('.')[0], temp)
                except Exception as e:
                    logger.error("MQTT exception: {}".format(e))
                    restart()
        else:
            print("NO data to process")
        time.sleep(5)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Pipeline for the car plate and number detection"
    )
    setup_logging()
    logger = logging.getLogger(__name__)
    parser.add_argument('command', metavar='<command>', help="''")

    parser.add_argument('--env', required=True,
                        help='Environment to choose')


    
    
    args = parser.parse_args()

    if args.env == 'bj':
        config = ConfigGlobalAPI()
    if args.env == 'sz':
        config = ConfigSuzhouAPI()
    if args.env == 'hk':
        config = ConfigHKAPI()
    
    sftp = SFTpServer((config.FTP_host, config.FTP_port),config)
    try:
        mq = MQTTClient(config)
    except MqttDisconnectException as e:
        logger.error("EXCEPTION RAISED: {}".format(e))




    if args.command == 'cron':
        try:
            cron_job(mqtt_client=mq, logger=logger, sftp=sftp)

        except KeyboardInterrupt:
            print("Exception")


