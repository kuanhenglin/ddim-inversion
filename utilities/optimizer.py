from torch import optim


def get_optimizer(name, parameters, learning_rate, weight_decay,
                  beta_1, amsgrad=False, epsilon=1e-7, maximize=False):
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay,
                               betas=(beta_1, 0.999), amsgrad=amsgrad, eps=epsilon,
                               maximize=maximize)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay,
                                  maximize=maximize)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, maximize=maximize)
    else:
        raise NotImplementedError(f"Optimizer {name} not supported.")
    return optimizer
