import numpy as np
from cart.utils import square_loss, rmse, param_grid


class grid_search_cross_validation(object):
    """grid search cross validation class
    Args:
        model_class: model class with a fit and predict function
        parameter_dict: a dictionary of parameter names and their list of options
        k: number of folds
        loss_metric: metric used to evaluate the model's performance. Options are `square_loss` and `rmse`. 
    """

    def __init__(self, model_class, parameter_dict, k=5, loss_metric="square_loss"):
        self.k = k
        self.loss_metric = eval(loss_metric)
        self.model_class = model_class
        self.best_models = []
        self.mean_training_losses = []
        self.mean_validation_losses = []
        self.optimal_model = None
        self.parameter_grid = param_grid(parameter_dict)

    def fit(self, x, y):
        """Complete k fold cross validation
        Args:
            x: nxd matrix of training data
            y: n vector of training labels
        Returns:
            self.optimal_model: the optimal model evaluated using k fold cross validation       
        """
        indices = self._generate_kFold(len(x))
        self._cross_validation(x, y, indices)
        return self.optimal_model

    def _generate_kFold(self, n):
        """Generate k folds
        Input:
            n: number of training examples
        Returns:
            kfold_indices: a list of len k. Each entry takes the form
            (train indices, val indices)
        """
        assert self.k >= 2
        kfold_indices = []

        idx = np.array(range(n))
        folds = np.array(np.array_split(idx, self.k))
        folds_idx = np.array(range(self.k))
        val_idxs = np.random.choice(len(folds), len(folds), replace=False)

        for val_idx in val_idxs:
            val_idx_bool = np.zeros(self.k, dtype=bool)
            val_idx_bool[val_idx] = True
            train_fold_idx = folds_idx[~val_idx_bool]
            val_fold_idx = folds_idx[val_idx_bool]
            train_fold = []
            for i in train_fold_idx:
                train_fold = train_fold + list(folds[i])

            val_fold = []
            for i in val_fold_idx:
                val_fold = val_fold + list(folds[i])

            kfold_indices.append((train_fold, val_fold))

        return kfold_indices

    def _cross_validation(self, xTr, yTr, indices):
        """ k-fold cross validation
        Args:
            xTr: nxd matrix (training data)
            yTr: n vector (training data)
            indices: indices from generate_kFold
        """
        training_losses = []
        validation_losses = []
        num_parameters = len(self.parameter_grid)
        k_fold_train_losses = np.ones((num_parameters, len(indices)))
        k_fold_val_losses = np.ones((num_parameters, len(indices)))

        for i, index in enumerate(indices):
            xTr_i = xTr[(index[0])]
            yTr_i = yTr[(index[0])]
            xVal_i = xTr[(index[1])]
            yVal_i = yTr[(index[1])]
            best_model, train_losses, val_losses = self._grid_search(
                xTr_i, yTr_i, xVal_i, yVal_i
            )
            k_fold_train_losses[:, i] = train_losses
            k_fold_val_losses[:, i] = val_losses
            self.best_models.append(best_model)

        self.mean_training_losses = np.mean(k_fold_train_losses, axis=1)
        self.mean_validation_losses = np.mean(k_fold_val_losses, axis=1)
        parameters_min_val_loss = np.where(
            self.mean_validation_losses == np.array(self.mean_validation_losses).min()
        )[0][0]
        optimal_parameters = self.parameter_grid[parameters_min_val_loss]
        # check which model contains the optimal parameters
        optimal_model_bool = np.array(
            [
                optimal_parameters.items() <= best_model.__dict__.items()
                for best_model in self.best_models
            ]
        )
        optimal_model_index = np.where(optimal_model_bool)[0][0]
        self.optimal_model = self.best_models[optimal_model_index]

    def _grid_search(self, xTr, yTr, xVal, yVal):
        """ Grid search to find the best parameters that minimise the validation loss
        Args:
            xTr: nxd matrix
            yTr: n vector
            xVal: mxd matrix
            yVal: m vector
        Returns:
            best_model: the model that yields that lowest loss on the validation set
            training losses: a list of len k. the i-th entry corresponds to the the training loss
                    the model of parameter_grid[i]
            validation_losses: a list of len k. the i-th entry corresponds to the the validation loss
                    the model of parameter_grid[i]
        """
        training_losses = []
        validation_losses = []

        best_val_loss = np.inf

        for params in self.parameter_grid:
            model = self.model_class(**params)
            model.fit(xTr, yTr)
            training_losses.append(self.loss_metric(model.predict(xTr), yTr))
            val_loss = self.loss_metric(model.predict(xVal), yVal)
            validation_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        return best_model, training_losses, validation_losses
