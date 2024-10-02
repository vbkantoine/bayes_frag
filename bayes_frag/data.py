# data script
# dedicated to get the data and deliver them in a convenient format
import os
import numpy as np
import numpy.random as rd
import pandas as pd
import matplotlib.pyplot as plt

from config import IM_dict, data_path, csv_path

def get_data(path, csv_path) :
    """get as a dataframe the csv-data"""
    return pd.read_csv(os.path.join(path,csv_path))

class Data() :
    """"
    Data class. Contains full dataset and drawn methods for it.

    Attributes:
        df: the full input data from a csv file.
        shuffle_ids: permutation of the indices from the raw data and the stored data.
        IM: name of the considered IM.
        A: stored IMs from the data.
        Y: stored responses from the data.
        C: threshold to characterize a failure.
        Z: stored failures from the data.
        increasing_mode: if True, default drawn from the dataset will not be random.
        a_tab, h_a, f_A: serve the computation of an approximated density of the IM.
    """
    def __init__(self, IM, shuffle=False, full_size=None, quantile_C=None, C=None, path=data_path, csv_path=csv_path, name_inte='z_max') :
        """
        Extract data from a CSV file with a column [IM] and a column of responses [name_inte] at least.
        A threshold C serves the computation of a failures array: failure==(data[name_inte]>C).

        Args:
            IM (string): IM name to get characteristic from config
            shuffle (bool, optional): True if the dataset must be shuffled at initialization. Defaults to False.
            full_size (_type_, optional): not used. Defaults to None.
            quantile_C (float, optional): if not None, C will be set such that (100*(1-quantile_C))% of data result into a failure. Defaults to None.
            C (float, optional): to set manually a threshold value to define failures/non-failures. Ignored if quantile_C is not None. Defaults to None.
                If quantile_C==C==None, C is taken from config.
            path (string, optional): path to the data folder. Defaults to data_path.
            csv_path (string, optional): path to the csv file from data_path. Defaults to csv_path.
            name_inte (string, optional): column name of the structure's rsponse. Defaults to 'z_max'.
        """
        self.df = get_data(path, csv_path)
        self.shuffle_ids = self._shufflize(shuffle)
        self.A = self.df[[IM]].values[self.shuffle_ids]
        self.Y = self.df[[name_inte]].values[self.shuffle_ids]
        if quantile_C is None :
            if C is None :
                self.C = IM_dict[IM]['C']
            else :
                self.C = C
        else :
            self.C = np.quantile(self.Y, quantile_C)+1e-6
        self.Z = 1*(self.Y>=self.C)
        self.IM = IM
        self.a_tab = None
        self.h_a = None
        self._set_a_tab()
        self.f_A = None
        self.f_A_tab = None
        self._compute_f_A()
        self.increasing_mode = True

    def _shufflize(self, shuffle) :
        ids = np.arange(len(self.df))
        if not shuffle :
            return ids
        else :
            np.random.shuffle(ids)
            return ids


    def set_increasing_mode(self) :
        """Set self.increasing_mode to True
        """
        self.increasing_mode = True

    def set_not_increasing_mode(self) :
        """Set self.increasing_mode to False
        """
        self.increasing_mode = False

    def default_draw(self, k) :
        """Draw k data.
            Returns self.increasing_draw(k) if self.increasing_mode is set to True.
            Returns self.draw(k) otherwise.
        Args:
            k (int): dataset-size expected.
        """
        if self.increasing_mode :
            return self.increasing_draw(k)
        else :
            return self.draw(k)

    def draw(self, k) :
        """output a random drw of k items from the dataset
        Return A,Y,Z"""
        ids = rd.randint(self.A.shape[0], size=k+1)
        return self.A[ids], self.Y[ids], self.Z[ids]

    def increasing_draw(self, k) :
        """output the k first items of the dataset
        Return A,Y,Z"""
        return self.A[:k+1], self.Y[:k+1], self.Z[:k+1]

    def _set_a_tab(self, min_a=10**-5, max_a=None, num_a=200) :
        """compute a_tab and update self.a_tab accordigly"""
        if max_a is None :
            max_a = self.A.max()
        self.a_tab, self.h_a = np.linspace(min_a, max_a, num=num_a, retstep=True)

    def _compute_f_A(self, sigma=None, h=None) :
        """compute approximated f_A and update self.f_A_tab accordingly"""
        n = len(self.A)
        if sigma is None :
            sigma = np.sqrt(self.A.var())
        if h is None :
            h = sigma*n**(-1/5)
        self.f_A = lambda a_im : np.exp(-((self.A[...,np.newaxis]-a_im[np.newaxis])/h)**2).mean(axis=0)/h/np.sqrt(np.pi)
        self.f_A_tab = self.f_A(self.a_tab).flatten()

    def plot_f_A(self, ax=None) :
        """plot approximated f_A. Create new figure if ax is None"""
        if ax is None :
            fig = plt.figure(figsize=(4.5,4))
            ax = fig.add_subplot(111)
        ax.plot(self.a_tab, self.f_A_tab, label=r'approx. $f_A$')
        ax.hist(self.A, 100, density=True, label=r'A hist.')
        ax.legend()
        ax.set_xlabel(r'$a$={}'.format(self.IM))
        ax.set_ylabel(r'$f_A(a)$')




# def data_draw() : # a draw in the data
#     return True

if __name__=="__main__":
    IM = 'PGA'
    data = Data(IM)

    plt.ion()
    plt.show()
    data.plot_f_A()

