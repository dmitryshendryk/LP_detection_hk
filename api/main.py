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


def cron_job(sftp_server, mqtt_client, logger):
    while True:
        try:
            file_list = sftp_server.sftp_download(REMOTE_FOLDER_PATH)
            print(file_list)
            sftp_server.sftp_remove(REMOTE_FOLDER_PATH, file_list)
        except Exception as e:
            logger.error("SFTP exception : {}".format(e))
            restart()
            
        imgs_path = os.path.join(ROOT_DIR,'imgs')
        files_path = os.path.join(ROOT_DIR, 'imgs/*')
        results=process(lp_model,char_model,imgs_path, logger)
        files = glob.glob(files_path)
        for f in files:
            os.remove(f)
        for item in results:
            try:

                temp = {
                    "code": item['code'],
                    "message": item['message'],
                    "licencePlate": item['licencePlate'],
                    "vehicleType": '',
                    "vehicleColor":"color"}
            except KeyError:
                continue
            try:
                print("recognition/vehicle/res/" + item['img_name'].split('.')[0])
                mqtt_client.publish("recognition/vehicle/res/" + item['img_name'].split('.')[0], temp)
            except Exception as e:
                logger.error("MQTT exception: {}".format(e))
                restart()
      
        time.sleep(10)

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
    
    server = SFTpServer((config.FTP_host, config.FTP_port),config)
    try:
        mq = MQTTClient(config)
    except MqttDisconnectException as e:
        logger.error("EXCEPTION RAISED: {}".format(e))



    if args.command == 'download':
        try:    
            server.sftp_download(REMOTE_FOLDER_PATH, connection_close=True)
        except KeyboardInterrupt:
            server.sftp.close()
            server.transport.close()
        
    if args.command == 'remove':
        try:
            server.sftp_remove(REMOTE_FOLDER_PATH, file_list=None, connection_close=True)
        except KeyboardInterrupt:
            server.sftp.close()
            server.transport.close()

    if args.command == 'cron':
        try:
            cron_job(sftp_server=server, mqtt_client=mq, logger=logger)

        except KeyboardInterrupt:
            server.sftp.close()
            server.transport.close()


