from torch import optim


def get_optimizer(name, parameters, learning_rate, weight_decay,
                  beta_1, amsgrad=True, epsilon=1e-7):
    optimizer = None
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay,
                               betas=(beta_1, 0.999), amsgrad=amsgrad, eps=epsilon)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {name} not supported.")
    return optimizer
