# reference_curves script
# compute reference curves estimation from a full dataset
import numpy as np
from scipy.special import erf
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from stat_functions import get_opp_log_vr_probit

##
#
class Reference_curve() :
    """Reference_curve class
    gather 'reference' items for comparison, computed with the consideration of all the available data"""
    def __init__(self, data) :
        """data = data object containing the dataset"""
        self.data = data
        opp_log_vraiss = get_opp_log_vr_probit(data.Z, data.A)
        self.theta_MLE = None
        self.curve_MLE = None
        self._compute_MLE_curve()
        self.reglin = LinearRegression().fit(np.log(data.A), np.log(data.Y))
        self.curve_MC = None
        self.curve_MC_var = None
        self.a_tab_MC = None
        self._compute_empirical_curves()

    def _compute_empirical_curves(self, Nc=35) :
        """compute empirically some fragility curves via MCMC and clustering"""
        # # below is depreciated
        # num_a = len(self.data.a_tab)
        # F_tab = np.zeros(num_a)
        # FF_tab = np.zeros(num_a)
        # for i,a in enumerate(self.data.a_tab[:-1]) :
        #     A_in_id = (self.data.A>=a)*(self.data.A<self.data.a_tab[i+1])
        #     fm = self.data.Z[A_in_id]
        #     FF_tab[i:] += fm.sum()
        #     if len(fm)==0 :
        #         F_tab[i] = 0
        #     else :
        #         F_tab[i] = fm.mean()
        # FF_tab /= self.data.Z.shape[0]
        # F_tab[-1] = 1
        # F_tab_lis = F_tab * np.exp(-(self.data.a_tab[np.newaxis]-self.data.a_tab[...,np.newaxis])**2/(2*self.data.h_a**2)).sum(axis=0)
        # F_tab_lis = F_tab_lis / F_tab_lis.max()
        # self.curve_MC, self.curve_MC_lis = F_tab, F_tab_lis
        # return F_tab, F_tab_lis
        kmeans_IM = KMeans(n_clusters=Nc).fit(self.data.A)
        a_m_tab = kmeans_IM.cluster_centers_.flatten()
        pMC = np.zeros_like(a_m_tab)
        sMC = np.zeros_like(a_m_tab)

        for i in range(Nc) :
            z_ci = self.data.Z[kmeans_IM.labels_==i]
            pMC[i] = np.mean(z_ci)
            sMC[i] = np.sqrt(pMC[i]*(1-pMC[i])/z_ci.shape[0])*1.96 #re-think the 1.96 constant w.r.t a desired alpha-confidence
        id_sort = np.argsort(a_m_tab)
        self.a_tab_MC = a_m_tab[id_sort]
        self.curve_MC = pMC[id_sort]
        self.curve_MC_var = sMC[id_sort]
        return kmeans_IM


    def _compute_MLE_curve(self, opp_log_vraiss=None) :
        if opp_log_vraiss is None :
            opp_log_vraiss = get_opp_log_vr_probit(self.data.Z, self.data.A)
        self.theta_MLE = optimize.minimize(opp_log_vraiss, np.array([3,0.3]), options={'maxiter':50}).x
        self.curve_MLE = 1/2 + 1/2 * erf(np.log(self.data.a_tab/self.theta_MLE[0])/self.theta_MLE[1])
        return self.theta_MLE

    def plot_ref_fig(self, ax=None, num_points=500) :
        """plot the reference fragility curves
            If ax is None a new figure is created"""
        if ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # ax.plot(self.data.a_tab, self.curve_MC, label='MC')
        ax.plot(self.a_tab_MC, self.curve_MC, '--', label='MC', color='grey', alpha=0.35)
        ax.fill_between(self.a_tab_MC, self.curve_MC+self.curve_MC_var, self.curve_MC-self.curve_MC_var, color='grey', alpha=0.6)
        # ax.plot(self.data.a_tab, self.curve_MC_lis, label='Kernel')
        ax.plot(self.data.a_tab, self.curve_MLE, label=r'MLE', color='magenta') #, $\theta=({:4.2f},{:4.2f})$'.format(self.theta_MLE[0], self.theta_MLE[1]))
        id_points = np.random.choice(np.arange(len(self.data.A)), size=num_points)
        ax.plot(self.data.A[id_points], self.data.Z[id_points], 'x', color='red', markersize=3, label='obs.')
        ax.set_xlim((self.data.a_tab.min(), self.data.a_tab.max()))
        ax.set_xlabel('a={} (m/s$^2$)'.format(self.data.IM))
        ax.set_ylabel(r'$P_f(a)$')
        ax.set_title(r'Reference fragility curve')
        ax.legend(loc='lower right')
        ax.set_ylim((-0.01, 1.01))
        return ax




if __name__=="__main__":
    from data import Data
    plt.ion()
    plt.show()
    IM = 'PGA'
    dat = Data(IM)
    ref = Reference_curve(dat)
    ref.plot_ref_fig()



