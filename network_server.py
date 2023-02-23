import socket
from PyQt5 import QtCore

class NetworkServer(QtCore.QThread):

    server_started_signal = QtCore.pyqtSignal(bool)
    server_ip_signal = QtCore.pyqtSignal(str)
    server_port_signal = QtCore.pyqtSignal(int)
    data_received_signal = QtCore.pyqtSignal(str)

    def __init__(self, local_ip, port):
        super(NetworkServer, self).__init__()

        self.local_ip = local_ip
        self.port = port

        self.server_is_running = True
        self.started.connect(lambda: self.set_server_started(True, local_ip, port))
        self.finished.connect(lambda: self.set_server_started(False, local_ip, port))

    def set_server_started(self, value, ip_address, port):
        self.server_started_signal.emit(value)
        self.server_ip_signal.emit(ip_address)
        self.server_port_signal.emit(port)

    def run(self):

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.local_ip, self.port)

        server_sock.bind(server_address)
        server_sock.listen(1)

        self.set_server_started(True, self.local_ip, self.port)
        while self.server_is_running:
            connection, _ = server_sock.accept()

            while True:
                try:
                    data = connection.recv(1024)
                except socket.error:
                    break
                if len(data) == 0:
                    break
                self.data_received_signal.emit(data.decode())

            connection.close()

        self.set_server_started(False, self.local_ip, self.port)
        server_sock.close()

    def stop(self):
        self.server_is_running = False
