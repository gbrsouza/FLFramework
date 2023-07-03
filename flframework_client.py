import flwr as fl

class FlFrameworkClient(fl.client.NumPyClient):

    def __init__(self, model, train_ds, epochs):
        super().__init__()
        self.model = model
        self.train_ds = train_ds
        self.epochs = epochs

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_ds, epochs=self.epochs)
        return self.model.get_weights(), self.epochs, {}

    def evaluate(self, parameters, coonfig):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.train_ds)
        return loss, self.epochs, {"accuracy": accuracy}