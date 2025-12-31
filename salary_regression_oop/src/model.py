class Model:
    def __init__(self, x, y, theta0, theta1):
        self.x = x
        self.y = y
        self.theta0 = theta0
        self.theta1 = theta1

    def linear_estimator(self):
        return self.theta0 + self.theta1 * self.x
    
    