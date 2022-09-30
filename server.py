from src.simulator.server_app import ServerApp
from src.models.cnn import CNN
from tensorflow.keras.models import model_from_json

global_model = CNN('saved_models/cnn/global-model/checkpoint')
global_model.create_model()
global_model.model.load_weights("model.h5")

app = ServerApp(max_timeout=100,
                global_model_json=global_model.model.to_json(),
                weigths=global_model.model.get_weights())

app.start()
