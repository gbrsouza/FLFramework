from src.models.cnn import CNN
from src.actors.server import Server
from src.utils.evaluator import evaluate_model
from src.data.dataset import Dataset

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import logging as log   
log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

if __name__ == "__main__":
    # read dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # here we have an array of iamges paths 
    # 2 - Transform and split dataset
    dataset = Dataset((train_images, train_labels))
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(train_size=0.80, validation=False)


    c1 = CNN('saved_models/weigths/client0/checkpoint')
    c2 = CNN('saved_models/weigths/client1/checkpoint')
    c3 = CNN('saved_models/weigths/client2/checkpoint')
    global_model = CNN('')

    c1.load_model()
    c2.load_model()
    c3.load_model()

    # print(c3.get_weights())

    # models = [c1, c2]
    # for step, model in enumerate(models):
    #     logger.info('Metrics for client %s', step)
    #     acc ,pre, rec, matrix = evaluate_model(test_images, test_labels, model, class_names)
    #     logger.info('\n-------confusion matrix-------\n%s', matrix)
    #     logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)

    # c1 = [ 3.86401087e-01,  4.74481061e-02,  7.40105957e-02,
    #       -6.48960620e-02,  3.66595313e-02,  8.47733021e-02,
    #        1.61575556e-01, -5.81597388e-02,  1.02529772e-01,
    #        1.43539876e-01, -1.11005850e-01, -5.02536185e-02,
    #        1.00040533e-01, -2.76796035e-02,  6.73845485e-02,
    #        1.24235786e-01,  6.38519228e-02,  5.97785972e-02,
    #        1.18141726e-01,  8.89112875e-02, -1.58666119e-01,
    #        1.14179611e-01,  4.48835790e-02, -8.97079930e-02,
    #       -3.05804480e-02, -2.76930302e-01,  4.77504507e-02,
    #       -4.85457256e-02, -9.33370963e-02,  1.12026595e-01,
    #       -9.69270468e-02, -6.15292639e-02]

    # c2 = [-5.31809367e-02, -1.00784048e-01, -1.15439914e-01,
    #        5.65438047e-02, -5.38485274e-02,  1.94351841e-02,
    #        1.24812298e-01, -1.35341182e-01,  2.43783355e-01,
    #       -9.81553420e-02,  7.94038773e-02,  2.01582879e-01,
    #       -7.22175166e-02,  2.09239826e-01,  1.89819098e-01,
    #        2.90550366e-02, -2.60957535e-02,  6.48630038e-02,
    #       -5.90800159e-02, -7.55851762e-03, -1.49681017e-01,
    #        2.71270096e-01, -3.64042334e-02,  1.78985536e-01,
    #       -8.49597007e-02, -2.23290566e-02, -1.23403668e-01,
    #       -1.04361460e-01, -2.70638485e-02,  9.44570825e-02,
    #       -5.43411188e-02,  9.18327123e-02]

    # c3 = [ 0.08032423,  0.15416595,  0.04367926,  0.10477382,
    #       -0.08446445, -0.15521723,  0.03165078,  0.21733677,
    #        0.11146167,  0.01461935,  0.15039186,  0.28797007,
    #        0.02618519,  0.04027636, -0.01052053,  0.13441175,
    #       -0.05928034, -0.015302  , -0.07459065,  0.03894235,
    #       -0.01976378,  0.12575503,  0.19990355, -0.0715551 ,
    #        0.00573029,  0.00062398, -0.02602691,  0.10959005,
    #       -0.06684606,  0.0329559 , -0.10175946, -0.10578506]

    server = Server()
    weights = server.aggregate_models([c1, c2, c3])
    print(weights)

    acc ,pre, rec, matrix = evaluate_model(test_images, test_labels, global_model, class_names)
    logger.info('Metrics from aggregated model')
    logger.info('\n-------confusion matrix-------\n%s', matrix)
    logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)