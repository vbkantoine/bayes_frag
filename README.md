# Objective priors for Seismic fragility curves
Objective priors construction for bayesian estimation of seismic fragility curves
Define Bayesian model for seismic fragility curves, compute Jeffreys prior, run simulations.

Code in "code" folder is derpreciated, focus on "new code" folder from now on.

## Config

In config script must be adapted:
`data_path`: data folder path
`csv_path`: data csv file name
And store the parameters per IM
```python
IM_dict['IM'] = {
      'C': <failure threshold>, 
      'save_fisher_arr': <thet_arrays(amin, amax, bmin, bmax, num_a, num_b>,
      'save_fisher_path': <fisher_array_save_path>
  }
```


## Use code

### Data

Create a Data class item to gather the dataset:
```python
from data import Data

IM = '<IM>'
data = Data(IM)

print(data.A.shape) # full dataset
print(data.draw(100)) # draw 100 item in dataset

data.plot_f_A() # plot approximated density of IM
```

### Reference curves

Compute the reference 'ground truth' on the model.

```python
from reference_curves import Reference_curve

ref = Reference_curve(IM)

print(ref.theta_MLE) # reference MLE on full dataset
ref.plot_ref_fig() # plot reference figures
```

### Hasting Metropolis 

Different HM algorithms are proposed into the `stat_functions` module.
The HM classes turns one into a regular and convenient object.
```python
import numba

import stat_functions as stat_f
import distribution


z0 = np.ones(2)
pi = stat_f.log_Jeffreys_lin
pi_log = True
iter_HM = 15000
HM_func =  stat_f.adaptative_HM

HM_simulator = stat_f.HM_Estimator(HM_func, z0, pi, pi_log=pi_log, iter_HM=iter_HM)
HM_simulator.simulate(5000)  # run 5000 simulations

HM_simulator.analyse_HM([-1])  # plot usefull analysis metrics of last simulations


# Post_HM_Estimator for a posteriori simulations
log_likelihood = stat_f.log_vrais
@numba.jit
def log_post(s,a,t) :
  return log_likelihood(s,a,t) + pi(t)

HM_a_posteriori = stat_f.Post_HM_Estimator(HM_func, z0, log_post, pi_log=pi_log, iter=iter_HM)
```

### Model

Create a Bayesian model, which from all above would run, store, simulate, and estimate fragility curves from different methods.

```python
from model import Model

model = Model(pi, log_likelihood, data, HM_a_posteriori)

model.run_simulations([25,50], 5000, sim=['post', 'MLE']) # simulate 5000 theta for dataset size 25 and 50 using posterior and MLE methods
model.plot_simul([25,50]) # plot the simulation results
```



## Jeffreys prior computation

Jeffreys and fisher function under binary model are defined in the `fisher` module. As they require time, the code is constructed to consider storage of its compuation.
To compute and save a new fisher array of a particular IM, update first the values for `'save_fisher_arr'` and `'save_fisher_path'` into `IM_dict[IM]` in `config` script

```python
imort os

from save_fisher import save_fisher
from data import Data
import config
import fisher

IM = '<IM>'
data = Data(IM)

alpha_tab = config.IM_dict[IM]['save_fisher_arr'].alpha_tab
beta_tab = config.IM_dict[IM]['save_fisher_arr'].beta_tab
save_path = os.path.join(config.data_path, config.IM_dict[IM]['save_fisher_path'])

function = fisher.fisher_function("simpson", data)
I = save_fisher(save_path, function, alpha_tab, beta_tab)
```

This allow the `extract_saved_fisher.Approx_fisher` to run and output a Fisher/Jeffreys function from the saved array:
```python
from extract_saved_fisher import get_jeffreys_IM
import plot_functions
import config

IM = '<IM>'
jeff_func = get_jeffreys_IM(IM)

# plot it:
alpha_tab = config.IM_dict[IM]['save_fisher_arr'].alpha_tab
beta_tab = config.IM_dict[IM]['save_fisher_arr'].beta_tab

for i,a in enumerate(alpha_tab) :
  for j,b in enumerate(beta_tab) :
    array[i,j] = jeff_func(np.array([a,b]))
plot_functions.plot_bi_function(array, alpha_tab, beta_tab) 
```







