import numpy as np
import math
class Logistic:
    """
    A custom Logistic Regression classifier supporting binary and multiclass
    classification with flexible regularization, gradient descent strategies,
    and preprocessing utilities.

    Parameters
    ----------
    kind : str, optional (default="binary")
        Type of classification task ("binary" or "multiclass").
    reg : str, optional (default="L1")
        Regularization technique ("L1", "L2", "ElasticNet", or "None").
    gradient : str, optional (default="minibatch")
        Gradient descent strategy ("batch", "minibatch", or "sdc").
    optimizer : str, optional (default="adam")
        Optimization algorithm ("adam", "momentum", "rmsprop", "lion").
    alpha : float, optional (default=0.5)
        ElasticNet mixing parameter (L1 ratio).
    lamb : float, optional (default=0.5)
        Regularization strength (lambda).
    test_percent : float, optional (default=20)
        Percentage of data used for testing.
    Lr : float, optional (default=0.001)
        Initial learning rate.
    batch_size : int, optional (default=64)
        Number of samples per mini-batch.
    strategy : str, optional (default="mean")
        Missing value handling ("mean", "median", "mode", "const", "drop").
    const : float, optional (default=0.0)
        Value for 'const' imputation strategy.
    decay : str, optional (default="step")
        Learning rate decay type.
    decay_rate : float, optional (default=0.5)
        Factor for LR reduction.
    decay_step : int, optional (default=20)
        Epoch interval for step decay.

    Attributes
    ----------
    weights : np.ndarray
        Learned weight matrix.
    bias : np.ndarray
        Learned bias vector.
    X_mean : np.ndarray
        Feature means for normalization.
    X_std : np.ndarray
        Feature standard deviations for normalization.

    Examples
    --------
    >>> #example1
    >>> model = Logistic(kind="binary", reg="L2", gradient="minibatch", Lr=0.01)
    >>> X_train, X_test, y_train, y_test = model.split(X, y, seed=42, shuffle=True)
    >>> model.train(X_train, y_train, X_test, y_test, epochs=500)
    >>> accuracy = model.score(X_test, y_test)

    >>> #example2
    >>> model = Logistic(kind="binary", reg="L2", gradient="minibatch", Lr=0.01)
    >>> X = model.select_features(X,labels)
    >>> X = model.drop_features(X,labels)
    >>> X = model.label_encoder(X,labels)
    >>> X = model.clean(X)
    >>> X = model.normalization(X)
    >>> X_train, X_test, y_train, y_test = model.split(X, y, seed=42, shuffle=True)
    >>> model.train(X_train, y_train, X_test, y_test, epochs=500)
    >>> accuracy = model.score(X_test, y_test)
    >>> model.save(path)
    >>> model = Logistic_0.load(path)
    """
    def __init__(self, kind = "binary", reg = "L1",
                 gradient = "minibatch", optimizer = "adam",alpha = 0.5,
                 lamb = 0.5, test_percent = 20, Lr = 0.001,
                 batch_size = 64,strategy="mean", const = 0.0,
                 decay="step", decay_rate = 0.5,decay_step = 20):
        """
        The Architect: Sets the blueprints for how the model will learn and optimize.
        """
        self.kind = kind
        self.reg = reg
        self.gradient = gradient
        self.optimizer= optimizer
        self.beta1,self.beta2 = 0.9,0.99
        self.v_w,self.v_b = 0.0,0.0
        self.s_w,self.s_b = 0.0,0.0
        self.m_w = None
        self.m_b = None
        self.alpha = alpha
        self.lamb = lamb
        self.test_percent = test_percent
        self.iLr = Lr
        self.Lr = Lr
        self.eLr = 0.00001
        self.decay_rate = decay_rate
        self.deacy_step = decay_step
        self.batch_size = batch_size
        self.strategy = strategy
        self.const = const
        self.decay = decay
        self.X_std = None
        self.X_mean = None
        self.encoded_labels = []
        self.label_mapping = {}

    def summary(self):
        """
        The Dashboard: Prints a detailed report of the model's current settings,
        training status, and data preprocessing results.

        Parameters
        ----------
        None :
            This method does not take any external inputs; it reads directly
            from the model's internal attributes.

        Returns
        -------
        None :
            The function prints a formatted summary directly to the console
            and does not return any value.
        """
        print("=" * 50)
        print("📊 Logistic Regression Model Summary")
        print("=" * 50)

        # Model type
        print("\n🔹 Model Configuration")
        print(f"Type                : {self.kind}")
        print(f"Gradient Strategy   : {self.gradient}")
        print(f"Optimizer        : {self.optimizer}")
        if self.optimizer == "adam":
            print(f"  ↳ beta1 (momentum weight): {self.beta1}")
            print(f"  ↳ beta2 (scaling weight): {self.beta2}")
        if self.optimizer == "rmsprop":
            print(f"  ↳ beta2 (scaling weight): {self.beta2}")
        if self.optimizer == "lion" or self.optimizer == "momentum":
            print(f"  ↳ beta1 (momentum weight): {self.beta1}")

        print(f"Regularization      : {self.reg}")

        if self.reg == "ElasticNet":
            print(f"  ↳ Alpha (L1 ratio): {self.alpha}")
        if self.reg != "None":
            print(f"  ↳ Lambda          : {self.lamb}")

        # Training setup
        print("\n⚙️ Training Setup")
        print(f"Initial LR          : {self.iLr}")
        print(f"Current LR          : {self.Lr}")
        print(f"Decay Strategy      : {self.decay}")
        print(f"Decay Rate          : {self.decay_rate}")
        print(f"Decay Step          : {self.deacy_step}")
        print(f"Batch Size          : {self.batch_size}")
        print(f"Test Split          : {self.test_percent}%")

        # Preprocessing
        print("\n🧹 Preprocessing")
        print(f"Missing Value Strategy : {self.strategy}")
        if self.strategy == "const":
            print(f"  ↳ Constant Value     : {self.const}")

        if self.X_mean is not None and self.X_std is not None:
            print(f"Normalization       : Applied")
            print(f"  ↳ Mean (first 3)  : {self.X_mean[:3]}")
            print(f"  ↳ Std  (first 3)  : {self.X_std[:3]}")
        else:
            print(f"Normalization       : Not applied")

        # Encoding
        print("\n🔤 Encoding")
        print(f"Encoded Columns     : {self.encoded_labels}")
        print(f"Mappings Stored     : {len(self.label_mapping)}")

        print("\n" + "=" * 50)

    def clean(self,X):
        """
        The Data Janitor: Fills in missing values (NaNs) so the math doesn't break.

        Parameters
        ----------
        X : np.ndarray
            The raw feature matrix containing potential missing values.

        Returns
        -------
        X : np.ndarray
            A cleaned version of the matrix with all NaNs handled.
        """
        X = X.astype(float)
        if self.strategy == "drop":
            mask = ~np.isnan(X).any(axis = 1)
            X = X[mask]

        for col in range(X.shape[1]):
            nan_mask = np.isnan(X[:,col])
            if not nan_mask.any() :
                continue
            if self.strategy == "mean":
                value = np.nanmean(X[:,col])
            elif self.strategy == "median":
                value = np.nanmedian(X[:,col])
            elif self.strategy == "mode":
                vals,counts = np.unique(X[~nan_mask,col],return_counts=True)
                value = vals[np.argmax(counts)]
            elif self.strategy == "const":
                value = self.const
            else:
                return "wrong strategy"

            X[nan_mask,col]=value
        return X

    def label_encoder(self,X,labels):
        """
        The Translator: Converts categorical text into numeric IDs for the model.

        Parameters
        ----------
        X : np.ndarray
            The dataset containing categorical columns.
        labels : list[int]
            List of column indices that need to be encoded.

        Returns
        -------
        X : np.ndarray
            Dataset with strings replaced by integers.
        """
        if not self.encoded_labels:
            for col in labels:
                categories = list(dict.fromkeys(X[:,col]))
                mapping = {val:key for key,val in enumerate(categories)}
                self.label_mapping[col] = mapping
                X[:,col] = np.array([mapping[v] for v in X[:,col]])
            self.encoded_labels = labels
            return X
        else:
            for col in self.encoded_labels:
                X[:,col] = np.array([self.label_mapping[col][v] for v in X[:,col]])
            return X

    def _validator(self,X,y):
        X = np.asarray(X,dtype="object")
        y = np.asarray(y)
        if len(X) == 0:
            print("X_train is empty\n")
        if len(X) != len(y):
            print("no sample in x_train does not match with y_train\n")
        if self.kind == "multi" and y.ndim != 2 :
            print( "For multiclass, y must be 2D one-hot encoded. "
                "Use np.eye(n_classes)[labels] to convert.")

    def select_features(self,X,labels):
        """
        The VIP Lounge: Isolates only the specific columns you want the model to focus on.

        Parameters
        ----------
        X : np.ndarray
            The full feature matrix containing all available data.
        labels : list[int]
            The specific column indices you want to keep.

        Returns
        -------
        X_selected : np.ndarray
            A refined matrix containing only the requested feature columns.
        """
        return X[:,labels]

    def drop_features(self,X,labels):
        """
        The Bouncer: Removes unnecessary or noisy columns from your dataset to keep it clean.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix you wish to trim down.
        labels : list[int]
            The column indices of the features you want to discard.

        Returns
        -------
        X_reduced : np.ndarray
            A trimmed version of the matrix with the specified columns deleted.
        """
        return np.delete(X,labels,axis=1)

    def normalization(self,X):
        """
        The Great Equalizer: Scales features to have a mean of 0 and a standard deviation of 1.

        Parameters
        ----------
        X : np.ndarray
            Unscaled numeric feature matrix.

        Returns
        -------
        X : np.ndarray
            Standardized feature matrix.
        """
        self.X_mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
        mask = std==0
        std[mask]=1
        self.X_std = std
        X = (X-self.X_mean)/self.X_std
        return X

    def split(self,X,y,seed,shuffle=False):
        """
        The Partitioner: Divides data into training and testing sets.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target labels.
        seed : int
            Seed for reproducible random shuffling.
        shuffle : bool, optional (default=False)
            Whether to shuffle the data before splitting.

        Returns
        -------
        tuple : (X_train, X_test, y_train, y_test)
            Four numpy arrays representing the divided dataset.
        """
        self._validator(X,y)
        if shuffle :
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X))
            X,y = X[idx],y[idx]

        cut = int(len(X) * (1 - self.test_percent / 100))
        X_train,X_test = X[:cut],X[cut:]
        y_train,y_test = y[:cut],y[cut:]

        return X_train,X_test,y_train,y_test

    def _regularizationLoss(self):
        if self.reg == "L1":
            return self.lamb * np.sum(np.abs(self.weights))
        elif self.reg == "L2":
            return self.lamb * np.sum(self.weights ** 2)
        elif self.reg == "ElasticNet":
            return self.lamb * (self.alpha * np.sum(np.abs(self.weights)) + (1 - self.alpha) * np.sum(np.square(self.weights)))
        else:
            return 0

    def _Regularization(self):
        if self.reg == "L1":
            return self.lamb * np.sign(self.weights)
        elif self.reg == "L2":
            return  2 * self.lamb * self.weights
        elif self.reg == "ElasticNet":
            return self.lamb * (self.alpha * np.sign(self.weights)+ 2 * (1 - self.alpha) * self.weights)
        else:
            return 0

    def _learning_rate_decay(self,epochs,t):
        if self.decay == "step":
            self.Lr = self.iLr * (self.decay_rate ** (t // self.deacy_step))
        elif self.decay == "time":
            self.Lr = self.iLr/(1 + self.decay_rate*t)
        elif self.decay == "exponential":
            self.Lr = self.iLr * np.exp(-self.decay_rate*t)
        elif self.decay == "cosine":
            self.Lr = self.eLr + 0.5*(self.iLr-self.eLr)*(1+math.cos(t/(epochs+1)*math.pi))
        elif self.decay == "const":
            self.Lr = self.Lr
        else:
            pass

    def _sigmoid(self,z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def _softmax(self,z):
        exp_z = np.exp(z-z.max(axis= 1, keepdims= True)) #softmax
        return exp_z/exp_z.sum(axis= 1, keepdims= True)

    def _loss_0(self,y,yp):
        return -np.mean(y * np.log(yp+1e-9) + (1 - y) * np.log(1 - yp+1e-9))

    def _loss_1(self,y,yp):
        return -np.mean(np.sum(y * np.log(yp+ 1e-9), axis=1))

    def _forward(self,X):
        return X @ self.weights + self.bias

    def _backward(self,X,y,yp,t):
        delta = yp - y
        dW = (X.T @ delta)/len(X)
        dB = np.mean(delta, axis=0, keepdims=True)
        eps = 1e-8

        if self.optimizer == "momentum":
            self.v_w = (self.beta1 * self.v_w) + (1 - self.beta1)*dW
            self.v_b = (self.beta1 * self.v_b) + (1 - self.beta1)*dB
            self.weights -= self.Lr*(self.v_w + self._Regularization())
            self.bias -= self.Lr*(self.v_b)

        elif self.optimizer == "rmsprop":
            self.s_w = (self.beta2 * self.s_w) + (1 - self.beta2)*(dW**2)
            self.s_b = (self.beta2 * self.s_b) + (1 - self.beta2)*(dB**2)
            self.weights -= self.Lr * (dW / (np.sqrt(self.s_w) + eps) + self._Regularization())
            self.bias -= self.Lr * dB / (np.sqrt(self.s_b) + eps)

        elif self.optimizer == "adam":
            self.v_w = (self.beta1 * self.v_w) + (1 - self.beta1)*dW
            self.v_b = (self.beta1 * self.v_b) + (1 - self.beta1)*dB
            self.s_w = (self.beta2 * self.s_w) + (1 - self.beta2)*(dW**2)
            self.s_b = (self.beta2 * self.s_b) + (1 - self.beta2)*(dB**2)
            v_w_corr = self.v_w / (1 - self.beta1**t)
            v_b_corr = self.v_b / (1 - self.beta1**t)
            s_w_corr = self.s_w / (1 - self.beta2**t)
            s_b_corr = self.s_b / (1 - self.beta2**t)
            self.weights -= self.Lr * (v_w_corr / (np.sqrt(s_w_corr) + eps) + self._Regularization())
            self.bias -= self.Lr * v_b_corr / (np.sqrt(s_b_corr) + eps)

        elif self.optimizer == "lion":
            if self.m_w is None:
                self.m_w = np.zeros_like(self.weights)
            if self.m_b is None:
                self.m_b = np.zeros_like(self.bias)
            update_w = np.sign(self.beta1 * self.m_w + (1 - self.beta1) * dW)
            update_b = np.sign(self.beta1 * self.m_b + (1 - self.beta1) * dB)
            self.weights -= self.Lr * (update_w + self._Regularization())
            self.bias -= self.Lr * update_b
            self.m_w = self.beta2 * self.m_w + (1 - self.beta2) * dW
            self.m_b = self.beta2 * self.m_b + (1 - self.beta2) * dB

        else:
            self.weights -= self.Lr*(dW + self._Regularization())
            self.bias -= self.Lr*(dB)

    def _getBatch(self,X,y):
        no_of_sample = len(X)
        batch_size = self.batch_size
        for i in range(0,no_of_sample,batch_size):
            yield X[ i : i + batch_size ], y[ i : i + batch_size ]

    def _init_weights(self,X,y):
        features = X.shape[1]
        if self.kind == "binary":
            self.weights = np.random.randn(features,1)*0.003
            self.bias = np.random.randn(1,1)*0.003
        else:
            classes = y.shape[1]
            self.weights = np.random.randn(features,classes)*0.003
            self.bias = np.random.randn(1,classes)*0.003

    def train(self,X,y,Xt,yt,epochs=1000):
        """
        The Boot Camp: Iteratively adjusts weights to minimize error on the training data.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
        Xt : np.ndarray
            Validation/Test features.
        yt : np.ndarray
            Validation/Test labels.
        epochs : int, optional (default=1000)
            Number of times to pass through the full dataset.

        Returns
        -------
        None : Updates internal weights and bias.
        """
        self._init_weights(X,y)
        best_loss = float('inf')
        patience = 7
        counter = 0
        for i in range(1,epochs+1):
            if self.gradient == "minibatch":
                for X_batch, y_batch in self._getBatch(X,y):
                    #forward
                    z = self._forward(X_batch)
                    if self.kind == "binary":
                        y_predict = self._sigmoid(z)
                        #loss
                        loss = self._loss_0(y_batch,y_predict) + self._regularizationLoss()
                    else:
                        y_predict = self._softmax(z)
                        #loss
                        loss = self._loss_1(y_batch,y_predict) + self._regularizationLoss()
                    #Backward
                    self._backward(X_batch,y_batch,y_predict,i)

            if self.gradient == "batch":
                z = self._forward(X)
                if self.kind == "binary":
                    y_predict = self._sigmoid(z)
                    #loss
                    loss = self._loss_0(y,y_predict) + self._regularizationLoss()
                else:
                    y_predict = self._softmax(z)
                    #loss
                    loss = self._loss_1(y,y_predict) + self._regularizationLoss()
                #Backward
                self._backward(X,y,y_predict,i)

            if self.gradient == "sdc":
                n_samples = len(X)
                for i_sample in range(n_samples):
                    X_i, y_i = X[i_sample : i_sample + 1],y[i_sample : i_sample + 1]
                    z = self._forward(X_i)
                    if self.kind == "binary":
                        yp_i = self._sigmoid(z)
                        #loss
                        loss = self._loss_0(y_i,yp_i) + self._regularizationLoss()
                    else:
                        yp_i = self._softmax(z)
                        #loss
                        loss = self._loss_1(y_i,yp_i) + self._regularizationLoss()
                    #Backward
                    self._backward(X_i,y_i,yp_i,i)

            if self.kind == "binary":
                yp = self.predict(Xt)
                current_loss = self._loss_0(yt,yp) + self._regularizationLoss()
            else:
                yp = self.predict(Xt)
                current_loss = self._loss_1(yt,yp) + self._regularizationLoss()

            if best_loss > current_loss :
                best_loss = current_loss
                counter = 0
            else:
                counter += 1

            if counter > patience :
                break

            self._learning_rate_decay(epochs,i)

    def predict(self,X):
        """
        The Oracle: Generates final class predictions for new input data.

        Parameters
        ----------
        X : np.ndarray
            New feature data to classify.

        Returns
        -------
        predictions : np.ndarray
            Array of predicted class labels (integers).
        """
        X = self.label_encoder(X,self.encoded_labels)
        X = (X - self.X_mean) / self.X_std
        z = X @ self.weights + self.bias
        if self.kind == "binary":
            return (self._sigmoid(z) >= 0.5).astype(int).ravel()
        return np.argmax(self._softmax(z), axis=1)
        
    def predict_proba(self,X):
        """
        The Predictor: Generates probability estimates for input data.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to be predicted.
    
        Returns
        -------
        probabilities : np.ndarray
            Predicted class probabilities. Returns a 1D array for binary 
            classification or a 2D array (n_samples, n_classes) for multiclass.
        """
        X = self.label_encoder(X,self.encoded_labels)
        X = (X - self.X_mean) / self.X_std
        z = X @ self.weights + self.bias
        if self.kind == "binary":
            return self._sigmoid(z)
        return self._softmax(z)
        
    def score(self,X,y):
        """
        The Report Card: Calculates the accuracy percentage of the model's predictions.

        Parameters
        ----------
        X : np.ndarray
            Features to evaluate.
        y : np.ndarray
            Ground truth labels.

        Returns
        -------
        accuracy : float
            The ratio of correct predictions to total samples.
        """
        preds = self.predict(X)
        if y.ndim == 2:
                y = np.argmax(y, axis=1)
        return np.mean(preds == y.ravel())

    def save(self,path):
        """
        The Time Capsule: Saves the entire model state to a file for later use.

        Parameters
        ----------
        path : str
            The file path where the model should be stored.

        Returns
        -------
        None
        """
        import pickle
        with open(path,"wb") as f:
            pickle.dump(self,f)

    @staticmethod
    def load(path):
        """
        The Resurrection: Loads a saved model instance from a file.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Returns
        -------
        model : Logistic_0
            A fully loaded and functional model instance.
        """
        import pickle
        with open(path,"rb") as f:
            return pickle.load(f)
