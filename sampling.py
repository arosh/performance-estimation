import json
import pandas
import stanutil
import scipy
import pickle

L = 200

df = pandas.read_csv('data.csv', index_col=0)
with open('names.json') as f:
    names = json.load(f)

model_code = '''
data {
    int N;
    int L;
    vector[N] D;
    int G[N, L];
}

parameters {
    vector<lower=0,upper=1>[L] q;
    real<lower=0> a0;
    real<lower=0> b0;

    vector[L] pf;
    real mu_pf;
    real<lower=0> sigma_pf;

    real<lower=0> gamma;
}

model {
    q ~ beta(a0, b0);
    a0 ~ cauchy(0, 2.0);
    b0 ~ cauchy(0, 0.64);

    pf ~ cauchy(mu_pf, sigma_pf);
    mu_pf ~ cauchy(0.455, 0.025);
    sigma_pf ~ cauchy(0, 0.14);

    gamma ~ cauchy(13.3, 0.03);

    for (i in 1:N) {
        for (j in 1:L) {
            if (G[i,j] == 1) {
                target += bernoulli_lpmf(1 | q[j]) + bernoulli_lpmf(1 | inv_logit(gamma * (pf[j] - D[i])));
            } else {
                target += log_sum_exp(
                    bernoulli_lpmf(0 | q[j]),
                    bernoulli_lpmf(1 | q[j]) + bernoulli_lpmf(0 | inv_logit(gamma * (pf[j] - D[i])))
                );
            }
        }
    }
}
'''

data = {}
data['N'] = len(df)
data['L'] = L
data['D'] = df['difficulty'] / 1000
data['G'] = df.iloc[:, 1:L+1]
init = lambda: {
    'q': [0.7] * L,
    'a0': 1.6,
    'b0': 0.55,
    'pf': [0.455] * L,
    'mu_pf': 0.455,
    'sigma_pf': 0.115,
    'gamma': 13.3,
}
stan_model = stanutil.stan_cache(model_code)
fit = stan_model.sampling(data=data, seed=0, init=init)

with open('stan_model_and_fit.pkl', 'wb') as f:
    pickle.dump(stan_model, f)
    pickle.dump(fit, f)
