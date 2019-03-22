import schedule
import time



class Scheduller():

    def __init__(self, time_run, date_type):

        self.time_run = time_run
        self.date_type = date_type

       
       
    
    def run(self, fn, sftp_server, mqtt_client):
        if (self.date_type == 'minutes'):
            schedule.every(self.time_run).minute.do(fn, sftp_server=sftp_server, mqtt_client=mqtt_client)
        elif (self.date_type == 'seconds'):
            schedule.every(self.time_run).seconds.do(fn, sftp_server=sftp_server, mqtt_client=mqtt_client)

        while 1:
            schedule.run_pending()
            time.sleep(1)


  
    