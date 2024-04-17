import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

#----------------------------------------------------------------------
# Plot the progression of histograms to kernels
# np.random.seed(1)
# N = 20
# # np.random.normal 画一个随机的normal（高斯）分布
# # np.concatenate 把多个arrays联合在一起
# X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
#                     np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
# X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
# bins = np.linspace(-5, 10, 10)

# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# fig.subplots_adjust(hspace=0.05, wspace=0.05)

# # histogram 1
# ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', density=True, stacked=True)
# ax[0, 0].text(-3.5, 0.31, "Histogram")

# # histogram 2
# ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', density=True, stacked=True)
# ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# # tophat KDE
# kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
# log_dens = kde.score_samples(X_plot)
# ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
# ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# # Gaussian KDE
# kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
# log_dens = kde.score_samples(X_plot)
# ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
# ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

# for ax in ax.ravel():
#     ax.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
#     ax.set_xlim(-4, 9)
#     ax.set_ylim(-0.02, 0.34)

# for ax in ax[:, 0]:
#     ax.set_ylabel('Normalized Density')

# for ax in ax[1, :]:
#     ax.set_xlabel('x')

#----------------------------------------------------------------------
# Plot all available kernels
X_plot = np.linspace(-6, 6, 1000)[:, None]
X_src = np.zeros((1, 1))

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)


def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x
#, 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'


log_dens = KernelDensity(kernel='gaussian').fit(X_src).score_samples(X_plot)
ax.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
ax.text(-2.6, 0.95, 'gaussian')

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.NullLocator())

ax.set_ylim(0, 1.05)
ax.set_xlim(-2.9, 2.9)

ax.set_title('Gaussian Kernels')
plt.savefig('./data/gaussion_kernel.png')
#----------------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.3 * N)),
                    np.random.normal(2, 1, int(0.4 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.3 * norm(5, 1).pdf(X_plot[:, 0])+ 0.4 * norm(2, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')

for kernel in ['gaussian']:#, 'tophat', 'epanechnikov'
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()
plt.savefig('./data/dataset.png')