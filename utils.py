import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interpolate
from sklearn import gaussian_process as gp


def generate_data(length_scale=0.2, x_min=0, x_max=1, num_feature=1000,
                  num_data=10000, num_sensor=100, s0=None):
    if s0 is None:
        s0 = [0]
    RBF = gp.kernels.RBF(length_scale=length_scale)
    X = np.linspace(x_min, x_max, num_feature)[:, None]
    K = RBF(X)  # 协方差矩阵
    L = np.linalg.cholesky(K + 1e-10 * np.eye(num_feature))  # 正定矩阵分解为上/下三角矩阵
    gp_samples = np.dot(L, np.random.randn(num_feature, num_data)).T  # 高斯采样

    def generate(gp_sample):
        u_fn = interpolate.interp1d(np.linspace(x_min, x_max, num=gp_sample.shape[-1]), gp_sample,
                                    kind='cubic', copy=False, assume_sorted=True)  # 插值法获取u(x)
        x = np.sort(np.random.rand(1)) * (x_max - x_min) + x_min
        y = solve_ivp(lambda t, y: u_fn(t), [x_min, x_max], s0, 'RK45', x).y[0]
        u_sensor = u_fn(np.linspace(x_min, x_max, num=num_sensor))
        return np.hstack([np.tile(u_sensor, (1, 1)), x[:, None], y[:, None]])  # 水平方向叠加（列增多）

    res = np.vstack(list(map(generate, gp_samples)))  # 垂直方向叠加
    return  res[..., :-1], res[..., -1:]