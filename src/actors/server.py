import tensorflow as tf

class Server():
    def __init__(self) -> None:
        pass

    def scale_model_weights(self, weight, scalar):
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final

    def sum_scaled_weights(self, scaled_weight_list):
        avg_grad = list()
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0).numpy()
            avg_grad.append(layer_mean)

        return avg_grad      

    def aggregate_models (self, models):
        """aggregate all models

        Args:
            models (Array): A array of models to aggregate
        """
        scaled_local_weight_list = list()
        scalar = 1.0/float(len(models))

        for model in models:
            # print("model before scalar")
            # print(model.get_weights()[0][0])
            scaled_weights = self.scale_model_weights(model.get_weights(), scalar)
            #scaled_weights = self.scale_model_weights(model, scalar)
            scaled_local_weight_list.append(scaled_weights)

        average_weights = self.sum_scaled_weights(scaled_local_weight_list)
        # print("model after aggregation")
        # print(average_weights[0][0])
        return average_weights