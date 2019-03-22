import paho.mqtt.client as mqtt 
import time
import json
import logging


class MQTTClient():

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.host = config.MQTT_broker_address
        self.port = config.MQTT_port
        self.client = mqtt.Client("", True, None, mqtt.MQTTv31)
        self.client.username_pw_set(config.MQTT_username,config.MQTT_password)
        self.client.connect(config.MQTT_broker_address, keepalive=config.MQTT_keep_alive, port=config.MQTT_port)

        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_disconnect = self.on_disconnect
        self.client.loop_start()
        self.logger.info("Start mqtt client")

    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info('Connection Ok received with code %d.' %(rc))
        else:
            self.logger.error('Connection failed with code %d' %(rc) )

    def on_publish(self, client, userdata, mid):
        self.logger.info("on_publish: "  + str(mid) )

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.logger.error('Unexpected disconnection')
        
    

    
    def publish(self, topic, data):
        try:
            rc, mid = self.client.publish(topic, json.dumps(data), qos=2)
            if rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info("Message {} queued successfully.".format(data))
            else:
                self.logger.error("Failed to publish message. Error: {}".format(result))
        except Exception as e:
            self.logger.error("EXCEPTION RAISED: {}".format(e))   
