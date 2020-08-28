import numpy as np


def square_impurity(y_train):
    """Compute the squared loss impurity of the labels
    
    Args:
        y_train: n-dimensional vector of labels
    
    Returns:
        square_impurity: squared loss impurity
    """
    (N,) = y_train.shape
    assert N > 0  # must have at least one sample

    impurity = np.sum(np.square(y_train - np.mean(y_train)))

    return impurity


def square_loss(pred, label):
    """Compute the mean squared loss
    
    Args:
        pred: numpy array of predictions
        label: numpy array of true labels
        
    Returns:
        squared loss based on predictions and true labels
    
    """
    return np.mean((pred - label) ** 2)


def rmse(pred, label):
    """Compute the root mean squared error
    
    Args:
        pred: numpy array of predictions
        label: numpy array of true labels
        
    Returns:
        root mean squared error based on predictions and true labels
    
    """
    return np.sqrt(square_loss(pred, label))


def train_test_split(x, y, train_size: float):
    xy = np.hstack((x, np.reshape(y, (len(y), -1))))
    train, test = np.array_split(xy, [int(train_size * len(xy))])
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    return x_train, x_test, y_train, y_test


def display_seconds(seconds: int, granularity: int = 1):
    """Display seconds as a string in hours, minutes and seconds
    
    Args:
        seconds: number of seconds to convert
        granularity: level of granularity of output
        
    Returns:
        string containing seconds in hours, minutes and seconds
    """

    intervals = {
        "hours": 3600,
        "minutes": 60,
        "seconds": 1,
    }
    result = []

    for name in intervals:
        count = intervals[name]
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name[:-1]
            result.append(f"{value} {name}")
    return ", ".join(result[:granularity])


def param_grid(grid: dict):
    """Convert a dictionary of parameter names and values to a parameter grid
    
    Args:
        grid: dictionary of parameter names (str) and values (list). Parameter values
          of type str are not allowed
        
    Examples:
        >>> param_grid({'train_steps': [500, 1000, 2000], 'regularisation':[True, False]})
        [{'train_steps': 500, 'regularisation': 1},
         {'train_steps': 500, 'regularisation': 0},
         {'train_steps': 1000, 'regularisation': 1},
         {'train_steps': 1000, 'regularisation': 0},
         {'train_steps': 2000, 'regularisation': 1},
         {'train_steps': 2000, 'regularisation': 0}]
    
    Returns:
        arr_dict: list of dictionaries containing all possible parameter combinations 
    
    """

    num_parameters = len(grid)
    parameter_names = np.array(list(grid.keys()))
    parameter_combination = np.array(np.meshgrid(*grid.values())).T.reshape(
        -1, num_parameters
    )

    arr_dict = []
    for params in parameter_combination:
        param_dict = {}
        for i, n in enumerate(parameter_names):
            param_dict[n] = params[i]
        arr_dict.append(param_dict)

    return arr_dict
