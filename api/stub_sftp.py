import time
import socket
import optparse
import sys
import textwrap
import os
import paramiko

from sftpserver.stub_sftp import StubServer, StubSFTPServer

import threading

HOST, PORT = 'localhost', 60022
BACKLOG = 10

class ConnHandlerThd(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self._conn = conn
        # self._keyfile = keyfile

    def run(self):
        # host_key = paramiko.RSAKey.from_private_key_file(self._keyfile)
        transport = paramiko.Transport(self._conn)
        # transport.add_server_key(host_key)
        transport.set_subsystem_handler(
            'sftp', paramiko.SFTPServer, StubSFTPServer)

        server = StubServer()
        transport.start_server(server=server)

        channel = transport.accept()
        while transport.is_active():
            print('Tick time')
            time.sleep(1)


def start_server(host, port, level):
    paramiko_level = getattr(paramiko.common, level)
    paramiko.common.logging.basicConfig(level=paramiko_level)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    server_socket.bind((host, port))
    server_socket.listen(BACKLOG)

    while True:
        conn, addr = server_socket.accept()

        srv_thd = ConnHandlerThd(conn)
        srv_thd.setDaemon(True)
        srv_thd.start()