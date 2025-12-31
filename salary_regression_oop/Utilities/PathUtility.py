import os

class PathFinder:
    def salary_regression_directory(self):
        return os.getcwd()
    def data_path(self):
        return os.path.join(self.salary_regression_directory(), 'data')
    def docs_path(self):
        return os.path.join(self.salary_regression_directory(), 'docs')   
    def images_path(self):
        return os.path.join(self.salary_regression_directory(), 'images') 
    def root_path(self):
        return os.path.dirname(self.salary_regression_directory())