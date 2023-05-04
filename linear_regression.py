'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Grace Moberg
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])
      
    
        if method == 'scipy':
            c = self.linear_regression_scipy(self.A, self.y)
        elif method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
        elif method == 'qr':
            c = self.linear_regression_qr(self.A, self.y)
        
        self.slope = c[1:] # slopes are all indices of c after first
        self.intercept = c[0,0] # intercept is first index of c
        

        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()
            

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        A = np.hstack([np.ones([A.shape[0], 1]), A]) # creates A matrix with h coord on left
        c, _, _, _ = scipy.linalg.lstsq(A,y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A = np.hstack([np.ones([A.shape[0], 1]), A])
        c = np.linalg.inv(np.matmul(A.T, A))@A.T@y
        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        A = np.hstack([np.ones([A.shape[0], 1]), A])
        Q,R = self.qr_decomposition(A)
        rhs = Q.T @ y
        c = scipy.linalg.solve_triangular(R, rhs)

        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''

        Q = np.ones([A.shape[0], A.shape[1]])
        i=0 # pointer to keep track of which column of A we are in
        j=0 # pointer to keep track of which column of Q we are in

        while i < (A.shape[1]):
            if i == 0:
                temp = A[:,i].copy()
                temp = temp/np.linalg.norm(temp)
                Q[:,i] = temp
                i += 1

            # if A.shape[1] == 1:
            #     break

            temp = A[:,i].copy()
            proj = 0

            while j < i:
                u = Q[:,j].copy()
                scalar = np.dot(u,temp)
                prod = u*scalar
                proj += prod
                j += 1
            
            orthT = temp - proj
                
            norm = np.linalg.norm(orthT)

            Q[:,i] = orthT/norm
            j = 0
            i+=1

        R = Q.T@A
    
        return Q,R


    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        
        if X is None: # checks if X is None and sets X to be self.A if it is
            X = self.A

        if self.p > 1:
            X = self.make_polynomial_matrix(X, self.p)
        
        y_pred = X@self.slope + self.intercept # computation

        return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        y_mean = self.mean([self.dep_var])
        y = self.data.select_data([self.dep_var])
        E = np.sum(np.square(y-y_pred))
        S = np.sum(np.square(y-y_mean))
        R2 = 1 - (E/S)

        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        resids = self.data.select_data([self.dep_var]) - y_pred
        return resids

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        N = len(self.residuals)
        mse = 1/N * np.sum(np.square(self.residuals))

        return mse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        x,y = analysis.Analysis.scatter(self, ind_var = ind_var, dep_var=dep_var, title = title)
        xMin = np.amin(x)
        xMax = np.amax(x)
        line_x = np.linspace(xMin, xMax, 1000)
        if self.p > 1:
            line_Mx = self.make_polynomial_matrix(line_x[:, np.newaxis], self.p)
            line_y = (line_Mx@self.slope) + self.intercept
        else: 
            line_y = (self.slope * line_x) + self.intercept
        plt.plot(line_x, np.squeeze(line_y.T))
        plt.title(label = (title  +', ' + ' R2 = ' + str(round(self.R2,5))))

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        
        fig, axs = super().pair_plot(data_vars, fig_sz)
       
        for i in range(len(data_vars)):
            for j in range (len(data_vars)):
                x = self.data.select_data([data_vars[j]])
                self.linear_regression([data_vars[j]],data_vars[i])
                xMin = np.amin(x)
                xMax = np.amax(x)
                line_x = np.linspace(xMin, xMax, 1000)
                line_y = (line_x * self.slope) + self.intercept
                axs[i,j].plot(line_x, np.squeeze(line_y.T), 'r')  
                axs[i,j].set_title(label = str(data_vars[i]) + ' vs ' + str(data_vars[j]) + ', R2=' + str(round(self.R2, 4)), fontsize = 'x-small')
                
                if hists_on_diag == True:
                    if i==j:
                        numVars = len(data_vars)
                        axs[i, j].remove()
                        axs[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
                        if j < numVars-1:
                            axs[i, j].set_xticks([])
                        else:
                            axs[i, j].set_xlabel(data_vars[i], fontsize = 'x-small')
                            axs[i,j].tick_params(labelsize = '8')
                        if i > 0:
                            axs[i, j].set_yticks([])
                        else:
                            axs[i, j].set_ylabel(data_vars[i], fontsize = 'x-small')
                            axs[i,j].tick_params(labelsize = '8')
                        axs[i,j].hist(x)
                        axs[i,j].set_title(label = 'histogram of ' + str(data_vars[j]), fontsize = 'x-small')
                        


    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        poly = np.ones([A.shape[0],p])
        for i in range(p):
            poly[:,i] = A[:,0].copy()**(i+1)
        return poly

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_vars = ind_var
        self.dep_var = dep_var
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])
        self.p = p

        poly = self.make_polynomial_matrix(self.A, self.p)
        self.A = poly 
        c = self.linear_regression_qr(self.A, self.y)
        
        self.slope = c[1:] # slopes are all indices of c after first
        self.intercept = c[0,0] # intercept is first index of c
        
        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])
       
        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()

    def cond(self, X=None):
        ''' Extension. Finds the matrix condition number using the equation
        2-norm(X) * 2-norm(X^-1) = cond.
        Matrices with large condition numbers make solution vectors c much more susceptible to errors.
        
        If X is not square, compute the Moore-Penrose pseudoinverse'''

        if X is None: # checks if X is None and sets X to be self.A if it is
            X = self.A
        if X.shape[0] != X.shape[1]:
            cond = np.linalg.norm(X)*np.linalg.norm(self.pseudoinverse(X))
        else:
            cond = np.linalg.norm(X)*np.linalg.norm(np.linalg.inv(X))
        return cond
    
    ''' The following functions are all part of an extension aimed at calculating the matrix condition number. 
    First, we need to be able to calculate the singular value decomposition of the matrix so that we can eventually calculate the
    pseudoinverse (so that a condition number may be calculated even if the number is not square.
    SVD = U@S@V.T'''
    
    def calcU(self, X=None):
        '''Helper method to calculate U.
        Returns ndarray U'''
        # columns of U are the eigenvectors X@X.T
        if X is None: # checks if X is None and sets X to be self.A if it is
            X = self.A
        newX = X@X.T
        eigenvalues, eigenvectors = np.linalg.eig(newX) # find eigenvalues and eigenvectors of matrix
        ncols = np.argsort(eigenvalues)[::-1] # sort eigenvalues 
        return eigenvectors[:,ncols]
    
    def calcV(self, X=None):
        ''' Helper method to calculate V.T. 
        Returns ndarray V.T.
        '''
        # columns of V are the eigvenvectors of X.T@X
        if X is None: # checks if X is None and sets X to be self.A if it is
            X = self.A
        newX = X.T @ X 
        eigenvalues, eigenvectors = np.linalg.eig(newX) # find eigenvalues and eigenvectors of matrix
        ncols = np.argsort(eigenvalues)[::-1] # sort eigenvalues
        return eigenvectors[:,ncols].T # transpose result in accordance with SVD formula
    
    def calcS(self, X=None):
        # diagonal matrix containing the square roots of the eigenvalues
        if X is None: # checks if X is None and sets X to be self.A if it is
            X = self.A
        if np.size(X@X.T) > np.size(X.T@X):
            newX = X.T@X
        else:
            newX = X@X.T
        eigenvalues, eigenvectors = np.linalg.eig(newX)
        eigenvalues = np.sqrt(eigenvalues)
        return eigenvalues[::-1] # sort eigenvalues in descending order
    
    def pseudoinverse(self, X=None):
        ''' Takes in U, S, V from helper methods. U and V are used to compute pseudoinverse.
        The pseudoinverse is given by A^+ = VD^+U.T, where V is V.T from calcV, U is the transpose of U,
        and D^+ is the pseudoinverse of S. Since S is diagonal, we can compute the pseudoinverse by
        creating a diagonal matrix from S, taking the reciprocal of each element, and transposing.
        
        Returns ndarray'''

        if X is None:
            X = self.A
        d = 1/self.calcS(X) # get reciprocals of S
        Vt = self.calcV(X)
        U = self.calcU(X)
        D = np.zeros(X.shape)
        if X.shape[0] < X.shape[1]:
            D[:X.shape[0],:X.shape[0]] = np.diag(d) # put reciprocals on diagonal
        else: 
            D[:X.shape[1],:X.shape[1]] = np.diag(d) # put reciprocals on diagonal

        P = Vt.T @ D.T @ U.T

        return P





