import os
import paramiko
from stat import S_ISDIR
import time
import requests 
import logging


ROOT_DIR = os.path.abspath('../')


class TimeLimitExceeded(Exception):
    pass


class SFTpServer(paramiko.Transport):

    def __init__(self, sock, config):
        super(SFTpServer, self).__init__(sock)
        self.logger = logging.getLogger(__name__)
        self.transport = paramiko.Transport((config.FTP_host, config.FTP_port))
        self.transport.connect(username = config.FTP_username, password = config.FTP_password)
        self.transport.packetizer.REKEY_BYTES = pow(2,40)
        self.transport.packetizer.REKEY_PACKETS = pow(2,40)
        self.transport.window_size=paramiko.common.MAX_WINDOW_SIZE
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
        
        self.logger.info("Start ftp client")


    def reconnect(self):
        self.sftp.close()
        self.transport.close()

   
    def sftp_walk(self, remotepath):
            path=remotepath
            files=[]
            folders=[]
            print(remotepath)
            for f in self.sftp.listdir_attr(remotepath):
                if S_ISDIR(f.st_mode):
                    folders.append(f.filename)
                else:
                    files.append(f.filename)
            if files:
                yield path, files


    def _timer(self,xfer,to_be_xfer):
        self.logger.info(" transfered: {0:.0f} %".format((xfer/to_be_xfer) * 100))
        # print(" transfered: {0:.0f} %".format((xfer/to_be_xfer) * 100))
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 60:
            self.logger.error('download timeout')
            raise TimeLimitExceeded

    def sftp_download(self, remotepath, connection_close=False):
        local_path = os.path.join(ROOT_DIR, 'imgs/')
        file_list = []
        for path,files  in self.sftp_walk(remotepath):
            self.logger.info(str(len(files)) + " images will be downloaded")
            # print(str(len(files)) + " images will be downloaded")
            count = 0
            for file in files:
                count += 1
                file_list.append(file)
                self.logger.info('Image name download: ' + file)
                self.logger.info(str(count) + '/' + str(len(files)))
                # print('Image name: ' + file)
                # print(str(count) + '/' + str(len(files)))
                try:
                    self.start_time = time.time()
                    self.sftp.get(os.path.join(os.path.join(path,file)), local_path + file, callback=self._timer)
                except TimeLimitExceeded as e:
                    raise Exception

                
        return file_list

        # print('end')
        if (connection_close):
            self.sftp.close()
            self.transport.close()
        
    
    def sftp_remove(self, remotepath, folder=None):
        # if file_list is not None:
        #     self.logger.info(str(len(file_list)) + " images will be deleted")
        #     # print(str(len(file_list)) + " images will be deleted")
        #     for file in file_list:
        #         self.sftp.remove(remotepath + "/" + file)
        # else:
        for path,files  in self.sftp_walk(remotepath + '/' + folder):
            self.logger.info(str(len(files)) + " images will be deleted")
            # print(str(len(files)) + " images will be deleted")
            for file in files:
                self.sftp.remove(remotepath + "/" + folder + "/"+ file) 

        self.sftp.rmdir(remotepath + '/' + folder)  
        

        # if (connection_close):
        #     self.sftp.close()
        #     self.transport.close()
        # print('end')


