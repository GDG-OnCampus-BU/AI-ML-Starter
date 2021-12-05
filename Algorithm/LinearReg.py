class LinearReg():

    def __init__(self,Method="Analytical"):
        """
        Linear Regression with 2 Different Method
        
        Param: Method -> "Analytical" or "GradientDes"
        """
        self.Method = Method
                    

        
    def fit(self,x,y,lr=0.0001,epoch=10000,iniM=1.5,iniB=5):
        """
        Prams:
        x = X data
        y = labels
        lr = Learning rate
        epoch = number of iteration
        iniM = initial slope(m)
        iniB = initial bias(b)
        
        Param: Method -> "Analytical" or "GradientDes"
        """    
        #Analytical Solution of Linear Regression (with MSE loss/LSO )
        if self.Method == "Analytical":
            self.theta = (x.T @ y) / (x.T @ x)
            
        #Gradient Descent Method    
        elif self.Method == "GradientDes":
            n = x.shape[0]
            self.m = iniM
            self.b = iniB
            for _ in range(epoch):
                b_gradient = -2 * sum(y - self.m*x + self.b) / n
                m_gradient = -2 * sum(x*(y - (self.m*x + self.b))) / n
                self.b = self.b + (lr * b_gradient)
                self.m = self.m - (lr * m_gradient)
        
    def predict(self,xnew):
        """
        Predict
        
        Param: 
        xnew = list or 1-D array X data for prediction
        """
        if self.Method == "Analytical":
            newy = self.theta * xnew
            return newy
        elif self.Method == "GradientDes":
            newy = self.m * xnew + self.b
            return newy