import pickle
import socket
import threading
import time
from src.actors.server import Server
from tensorflow.keras.models import model_from_json


class ServerApp(threading.Thread):

    def __init__(self, max_timeout=10, weigths=None, global_model_json=None):
        threading.Thread.__init__(self)
        self.max_timeout = max_timeout
        self.soc = socket.socket()
        self.connected = False
        self.received_data = None
        self.connection = None
        self.address = None
        self.weigths = weigths
        self.global_model_json = global_model_json

    def run(self):

        print(f"Iniciando servidor na porta 5454")
        self.soc.bind(('localhost', 5454))
        self.soc.listen(1)
        self.soc.settimeout(self.max_timeout)
        self.connection, self.address = self.soc.accept()
        self.connected = True
        self.start_server()
        self.soc.close()

    def start_server(self):
        data = {'model_name': 'global',
                'weigths': self.weigths,
                'model_json': self.global_model_json}
        response = pickle.dumps(data)
        self.connection.sendall(response)
        while True:
            received_data, status = self.receive_msg(8192)
            if status == 0:
                self.connection.close()

    def receive_msg(self, buffer_size):
        received_data = b''
        recv_start_time = time.time()
        while True:
            data = self.connection.recv(buffer_size)
            received_data += data
            if data == b'' and (time.time() - recv_start_time) > self.max_timeout:

                print(f"Servidor ecerrando por inatividade...")
                return None, 0

            elif str(data)[-2] == '.':

                print(f"Todos os dados recebidos pelo servidor, totalizando {len(data)} bytes.")
                received_data = pickle.loads(received_data)

                client_model = model_from_json(self.global_model_json)
                client_model.set_weights(received_data['weigths'])

                global_model = model_from_json(self.global_model_json)
                global_model.set_weights(self.weigths)

                models = [client_model, global_model]

                server = Server()
                weights = server.aggregate_models(models)

                global_model.set_weights(weights)
                global_model.save_weights("global_model.h5")

                print(f"Modelo global atualizado.")

                return None, 0
