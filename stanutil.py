def stanfit_to_dataframe(fit, pars=None):
    """
    Parameters
    ==========
    fit : pystan.StanFit4model
    """
    import pandas
    summary = fit.summary(pars=pars)
    columns = summary['summary_colnames']
    index = summary['summary_rownames']
    data = summary['summary']
    return pandas.DataFrame(data, index, columns)


def errorplot(data, x, y, error_low, error_high, hue=None):
    import seaborn
    import matplotlib.pyplot as plt

    def fn(d, label=None, color=None):
        err = [d[y] - d[error_low], d[error_high] - d[y]]
        plt.errorbar(x=d[x], y=d[y], yerr=err, fmt='o',
                     label=label, ecolor=color)

    if hue is not None:
        for label, color in zip(data[hue].unique(), seaborn.color_palette()):
            d = data[data[hue] == label]
            fn(d, label='%s = %s' % (hue, c), color=color)
    else:
        fn(data)

    xlim = plt.xlim()
    ylim = plt.ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    plt.plot([lim[0], lim[1]], [lim[0], lim[1]], 'k--')
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    if hue:
        plt.legend(loc='upper left')


def traceplot(fit, par, inc_warmup=False):
    """
    Parameters
    ==========
    fit : pystan.StanFit4model
    par : string
    inc_warmup : bool
    """
    import seaborn
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    alpha = 0.6
    # `pars` is ignored
    ms = fit.extract(permuted=False, inc_warmup=inc_warmup)
    trace = ms[:, :, fit.sim['fnames_oi'].index(par)]
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax1.set_title(par)
    for i in range(trace.shape[1]):
        ax1.plot(trace[:, i], alpha=alpha, label='chain%d' % (i + 1))
    ax1.legend(loc='best')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('value')
    ax2 = plt.subplot(gs[1], sharey=ax1)
    for i in range(trace.shape[1]):
        seaborn.kdeplot(trace[:, i], vertical=True, alpha=alpha, ax=ax2)
    # http://stackoverflow.com/a/32478701
    # ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(),
    # rotation='vertical') だとなぜか軸が消える
    for tick in ax2.get_xticklabels():
        # tick.set_rotation('vertical') にすると，思った方向と逆になる
        tick.set_rotation(-90)
    plt.tight_layout()


def stan_cache(model_code, dirname='stan_cache'):
    # http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html
    import os
    import pickle
    import pystan
    from hashlib import md5
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    code_hash = md5(model_code.encode('UTF-8')).hexdigest()
    cache_fn = os.path.join(dirname, '{}.pkl'.format(code_hash))
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    return sm


def MAP(fit, par):
    import scipy
    from scipy.stats import gaussian_kde
    ms = fit.extract(pars=par)[par]

    def func(ms_):
        # 極端に歪んだ分布でない限り，MAP推定値は95%信用区間の中には入っているだろう
        lo, hi = scipy.percentile(ms_, q=[2.5, 97.5])
        kde = gaussian_kde(ms_)
        xs = scipy.linspace(lo, hi, 1000)
        ys = kde.evaluate(xs)
        return xs[scipy.argmax(ys)]
    if len(ms.shape) == 1:
        return func(ms)
    else:
        retval = []
        n = ms.shape[1]
        for i in range(n):
            retval.append(func(ms[:, i]))
        return retval


def parse_advi(fit, par_regex=None):
    import pandas
    vb_sample = pandas.read_csv(
        fit['args']['sample_file'].decode('ascii'), comment='#')
    vb_sample = vb_sample.iloc[2:, :].reset_index(drop=True)
    if par_regex:
        return vb_sample.filter(regex=par_regex)
    else:
        return vb_sample
