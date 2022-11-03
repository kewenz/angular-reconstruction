import os
import glob
import yaml
import iminuit
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool

import mplhep as hep
hep.set_style(hep.styles.ROOT)

c = 299792458 / 1e9
GroundAltitude = 2100

Declination = 0
Inclination = 62.27
Inclination = np.pi / 2 + np.deg2rad(Inclination)
Bx = np.cos(Declination) * np.sin(Inclination)
By = np.sin(Declination) * np.sin(Inclination)
Bz = np.cos(Inclination)


def GetRefractionIndexAtXmax(x, y, z, Rs=325, Kr=0.1218):
    R_earth = 6370949.0
    h = np.sqrt((z + R_earth) ** 2 + x**2 + y**2) - R_earth
    Rh = Rs * np.exp(-Kr * h / 1e3)
    nh = 1 + Rh / 1e6
    return nh


# This is exactly what is computed in ZHAires
def GetZHSEffectiveactionIndex(
    x0, y0, z0, xant=0, yant=0, zant=0, ns=325, kr=-0.1218, stepsize=20000
):
    #      rearth=6371007.0 #new aires
    rearth = 6370949.0  # 19.4.0
    #     Variable n integral calculation ///////////////////
    R02 = (
        x0 * x0 + y0 * y0
    )  #!notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada (entro con injz-zXmax -> injz-z0=zXmax
    h0 = (
        np.sqrt((z0 + rearth) * (z0 + rearth) + R02) - rearth
    ) / 1e3  #!altitude of emission

    rh0 = ns * np.exp(kr * h0)  #!refractivity at emission (this
    n_h0 = 1 + 1e-6 * rh0  #!n at emission

    modr = np.sqrt(R02)

    if (
        modr > 1000
    ):  #! if inclined shower and point more than 20km from core. Using the core as reference distance is dangerous, its invalid in upgoing showers

        #         Vector from average point of track to observer.
        ux = xant - x0
        uy = yant - y0
        uz = zant - z0

        #         divided in nint pieces shorter than 10km
        nint = int((modr / stepsize) + 1)
        kx = ux / nint
        ky = uy / nint  # k is vector from one point to the next
        kz = uz / nint
        #
        currpx = x0
        currpy = y0  # current point (1st is emission point)
        currpz = z0
        currh = h0
        #
        sum = 0
        for iii in range(0, nint):
            nextpx = currpx + kx
            nextpy = currpy + ky  #!this is the "next" point
            nextpz = currpz + kz
            nextR2 = nextpx * nextpx + nextpy * nextpy
            nexth = (
                np.sqrt((nextpz + rearth) * (nextpz + rearth) + nextR2) - rearth
            ) / 1e3
            # c
            if np.abs(nexth - currh) > 1e-10:
                sum = sum + (np.exp(kr * nexth) - np.exp(kr * currh)) / (
                    kr * (nexth - currh)
                )
            else:
                sum = sum + np.exp(kr * currh)
            # c
            currpx = nextpx
            currpy = nextpy
            currpz = nextpz  #!Set new "current" point
            currh = nexth
        # c
        avn = ns * sum / nint
        n_eff = 1 + 1e-6 * avn  #!average (effective) n
    else:
        # c         withouth integral
        hd = zant / 1e3  #!detector altitude
        if np.abs(hd - h0) > 1e-10:
            avn = (ns / (kr * (hd - h0))) * (np.exp(kr * hd) - np.exp(kr * h0))
        else:
            avn = ns * np.exp(kr * h0)
        n_eff = 1 + 1e-6 * avn  #!average (effective) n
    return n_eff


class PWF:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def chi2(self, theta, phi, rc):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        # print('pars:', theta, phi, rc)
        sint, cost = np.sin(theta), np.cos(theta)
        sinp, cosp = np.sin(phi), np.cos(phi)

        dxx = self.x.reshape(1, -1) - self.x.reshape(-1, 1)
        dyy = self.y.reshape(1, -1) - self.y.reshape(-1, 1)
        dzz = self.z.reshape(1, -1) - self.z.reshape(-1, 1)
        dtt = self.t.reshape(1, -1) - self.t.reshape(-1, 1)
        fmin = dxx * sint * cosp + dyy * sint * sinp + dzz * cost - rc * c * dtt
        chi2 = (fmin**2).sum()
        """
        chi2 = 0
        for i, (xi, yi, zi, ti) in enumerate(zip(self.x, self.y, self.z, self.t)):
            for j, (xj, yj, zj, tj) in enumerate(zip(self.x, self.y, self.z, self.t)):
                if j <= i: continue
                # print(i, j, xi, xj, yi, yj, zi, zj, ti, tj)
                fmin = (xi-xj)*np.sin(theta)*np.cos(phi) + (yi-yj)*np.sin(theta)*np.sin(phi) + (zi-zj)*np.cos(phi) - rc*c*(ti-tj)
                # print(fmin)
                chi2 += fmin**2
        # print('chi2:', chi2)
        """
        return chi2

    def __call__(self, theta, phi, rc):
        chi2 = self.chi2(theta, phi, rc)
        # print(theta, phi, chi2)
        # exit(0)
        return chi2


class SWF:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def chi2(self, xs, ys, zs, rc):
        dx = self.x - xs
        dy = self.y - ys
        dz = self.z - zs

        # n = 1.0001
        n = GetZHSEffectiveactionIndex(xs, ys, zs, self.x.mean(), self.y.mean(), self.z.mean())
        rc = 1 / n

        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        ts = (self.t - dr / c / rc).mean()

        fmin = rc * c * (self.t - ts) - dr
        chi2 = (fmin**2).sum()
        # print(dr, ts, chi2)

        """
        chi2 = 0
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        ts = (self.t*c - dr).mean()
        for xi, yi, zi, ti in zip(self.x, self.y, self.z, self.t):
            dx = xi - xs
            dy = yi - ys
            dz = zi - zs
            fmin = rc*(ti*c-ts) - np.sqrt(dx**2+dy**2+dz**2)
            chi2 += fmin**2
        print(chi2)
        """
        return chi2

    def __call__(self, xs, ys, zs, rc):
        chi2 = self.chi2(xs, ys, zs, rc)
        return chi2


class ADF:
    def __init__(self, x, y, z, t, p, xs, ys, zs):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.p = p

        self.xs = xs
        self.ys = ys
        self.zs = zs

        n = GetRefractionIndexAtXmax(xs, ys, zs, Rs=325, Kr=0.1218)
        self.wc = np.arccos(1 / n)

    def chi2(self, theta, phi, dw, A, B, rc):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        # print('pars:', theta, phi, rc)

        sint, cost = np.sin(theta), np.cos(theta)
        sinp, cosp = np.sin(phi), np.cos(phi)

        kx = sint * cosp
        ky = sint * sinp
        kz = cost
        # print(kx, ky, kz)
        # print(kx**2 + ky**2 + kz**2)

        kBx = ky * Bz - kz * By
        kBy = kz * Bx - kx * Bz
        kBz = kx * By - ky * Bx
        norm = np.sqrt(kBx**2 + kBy**2 + kBz**2)
        kBx /= norm
        kBy /= norm
        kBz /= norm
        # print(kBx, kBy, kBz)
        # print(kBx**2 + kBy**2 + kBz**2)

        kkBx = ky * kBz - kz * kBy
        kkBy = kz * kBx - kx * kBz
        kkBz = kx * kBy - ky * kBx
        norm = np.sqrt(kkBx**2 + kkBy**2 + kkBz**2)
        kkBx /= norm
        kkBy /= norm
        kkBz /= norm
        # print(kkBx, kkBy, kkBz)
        # print(kkBx**2 + kkBy**2 + kkBz**2)

        # XmaxDist = (GroundAltitude - self.z) / kz
        # _asym = 0.01 * (1 - (kx * Bx + ky * By + kz * Bz) ** 2)

        xa = self.x - self.xs
        ya = self.y - self.ys
        za = self.z - self.zs
        # print(xa.shape, ya.shape, za.shape)

        xa_sp = xa * kBx + ya * kBy + za * kBz
        ya_sp = xa * kkBx + ya * kkBy + za * kkBz
        za_sp = xa * kx + ya * ky + za * kz
        # print(xa_sp.shape, ya_sp.shape, za_sp.shape)
        # exit(0)

        la = np.sqrt(xa**2 + ya**2 + za**2)
        eta = np.arctan2(ya_sp, xa_sp)
        omega = np.arccos(za_sp / la)
        alpha = np.arccos(kx * Bx + ky * By + kz * Bz)
        # print(la.shape, eta.shape, omega.shape, alpha)
        # print(min(eta), max(eta))
        # exit(0)

        # width = cost / np.cos(alpha) * dw
        # print(width)

        f_GeoM = 1 + 10**B * np.sin(alpha) ** 2 * np.cos(eta)
        f_Cerenkov = 1 / (
            1 + 4 * (((np.tan(omega) / np.tan(self.wc)) ** 2 - 1) / dw) ** 2
        )
        adf = 10**A / la * f_GeoM * f_Cerenkov
        # print(adf.shape)

        fmin = (self.p - adf)# / adf
        chi2 = (fmin**2).sum()
        # print(chi2)
        return chi2

    def __call__(self, theta, phi, dw, A, B, rc):
        chi2 = self.chi2(theta, phi, dw, A, B, rc)
        return chi2


def PWF_fit(x, y, z, ant_t, theta0, phi0):
    m = iminuit.Minuit(PWF(x, y, z, ant_t), theta=theta0, phi=phi0, rc=1)
    m.limits = ([90, 160], [0, 360], [0, 1])
    m.fixed = (False, False, True)
    m.errordef = 1
    m.tol = 0.0001
    # m.simplex()
    m.migrad()
    # m.hesse()
    # print(m)
    theta = m.values["theta"]
    phi = m.values["phi"]
    fval = m.fval
    return theta, phi, fval


def SWF_fit(x, y, z, ant_t, x0, y0, z0):
    m = iminuit.Minuit(SWF(x, y, z, ant_t), xs=x0, ys=y0, zs=z0, rc=1)
    m.limits = (
        [-1e5, 1e5],
        [-1e5, 1e5],
        [0, 1e5],
        [0, 1],
    )
    m.errors = (1, 1, 1, 0)
    m.fixed = (False, False, False, True)
    m.errordef = 1
    m.tol = 0.0001
    m.simplex()
    m.migrad()
    m.hesse()
    xs = m.values["xs"]
    ys = m.values["ys"]
    zs = m.values["zs"]
    fval = m.fval
    return xs, ys, zs, fval


def ADF_fit(x, y, z, ant_t, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, B0):
    m = iminuit.Minuit(
        ADF(x, y, z, ant_t, ant_p, xs, ys, zs),
        theta=theta0,
        phi=phi0,
        dw=dw0,
        A=A0,
        B=B0,
        rc=1,
    )
    m.limits = (
        [theta0 - 2, theta0 + 2],
        [phi0 - 2, phi0 + 2],
        [0.01, 5],
        [5, 10],
        [-6, 0],
        [0, 1],
    )
    # m.fixed = (True, True, True, False, False, True) # fixInfo_fixdw
    # m.fixed = (True, True, False, False, False, True) # fixInfo_freedw
    m.fixed = (False, False, True, False, False, True) # freeInfo_fixdw
    # m.fixed = (False, False, False, False, False, True) # freeInfo_freedw
    m.errordef = 1
    m.tol = 0.0001
    # m.simplex()
    m.migrad()
    m.hesse()
    # print(m)
    theta = m.values["theta"]
    phi = m.values["phi"]
    dw = m.values["dw"]
    A = m.values["A"]
    B = m.values["B"]
    fval = m.fval
    return theta, phi, dw, A, B, fval


def PWF_scan(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_p = data[:, 3]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_phi = shower_parameter[evtid, 0]
    pri_theta = shower_parameter[evtid, 1]
    pri_energy = shower_parameter[evtid, 2]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    thetaList = np.linspace(90, 160, 5)
    phiList = np.linspace(0, 360, 5)
    fmin = np.inf
    for theta0 in thetaList:
        for phi0 in phiList:
            theta, phi, fval = PWF_fit(x, y, z, ant_t, theta0, phi0)
            # print(theta, phi, fval)
            if fval < fmin:
                fmin = fval
                rec_theta = theta
                rec_phi = phi
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        fmin,
    )


def PWF_scan_withNoise(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_p = data[:, 3]

    np.random.seed(0)
    # dt = np.random.normal(scale=0, size=len(ant_t))
    # dt = np.random.normal(scale=1, size=len(ant_t))
    # dt = np.random.normal(scale=5, size=len(ant_t))
    dt = np.random.normal(scale=10, size=len(ant_t))
    # dt = np.random.normal(scale=15, size=len(ant_t))
    # dt = np.random.normal(scale=20, size=len(ant_t))
    ant_t += dt

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_phi = shower_parameter[evtid, 0]
    pri_theta = shower_parameter[evtid, 1]
    pri_energy = shower_parameter[evtid, 2]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])
    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    thetaList = np.linspace(90, 160, 5)
    phiList = np.linspace(0, 360, 5)
    fmin = np.inf
    for theta0 in thetaList:
        for phi0 in phiList:
            theta, phi, fval = PWF_fit(x, y, z, ant_t, theta0, phi0)
            # print(theta, phi, fval)
            if fval < fmin:
                fmin = fval
                rec_theta = theta
                rec_phi = phi
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        fmin,
    )


def SWF_scan(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_p = data[:, 3]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_phi = shower_parameter[evtid, 0]
    pri_theta = shower_parameter[evtid, 1]
    pri_energy = shower_parameter[evtid, 2]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    xList = np.linspace(-1e5, 1e5, 5)
    yList = np.linspace(-1e5, 1e5, 5)
    zList = np.linspace(0, 1e5, 5)
    fmin = np.inf

    for x0 in xList:
        for y0 in yList:
            for z0 in zList:
                x_xmax, y_xmax, z_xmax, fval = SWF_fit(x, y, z, ant_t, x0, y0, z0)
                # print(x_xmax, y_xmax, z_xmax, fval)
                if fval < fmin:
                    fmin = fval
                    rec_x_xmax = x_xmax
                    rec_y_xmax = y_xmax
                    rec_z_xmax = z_xmax
    rec_r_xmax = np.sqrt(
        (rec_x_xmax - pri_xcore) ** 2
        + (rec_y_xmax - pri_ycore) ** 2
        + (rec_z_xmax - GroundAltitude) ** 2
    )
    rec_theta = 180 - np.rad2deg(np.arccos((rec_z_xmax - GroundAltitude) / rec_r_xmax))
    rec_phi = 180 + np.rad2deg(
        np.arctan2(rec_y_xmax - pri_ycore, rec_x_xmax - pri_xcore)
    )
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_r_xmax,
        rec_x_xmax,
        rec_y_xmax,
        rec_z_xmax,
        fmin,
    )


def SWF_scan_withNoise(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_p = data[:, 3]

    np.random.seed(0)
    # dt = np.random.normal(scale=0, size=len(ant_t))
    # dt = np.random.normal(scale=1, size=len(ant_t))
    # dt = np.random.normal(scale=5, size=len(ant_t))
    dt = np.random.normal(scale=10, size=len(ant_t))
    # dt = np.random.normal(scale=15, size=len(ant_t))
    # dt = np.random.normal(scale=20, size=len(ant_t))
    ant_t += dt

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    xList = np.linspace(-1e5, 1e5, 5)
    yList = np.linspace(-1e5, 1e5, 5)
    zList = np.linspace(0, 1e5, 5)
    fmin = np.inf
    for x0 in xList:
        for y0 in yList:
            for z0 in zList:
                x_xmax, y_xmax, z_xmax, fval = SWF_fit(x, y, z, ant_t, x0, y0, z0)
                # print(x_xmax, y_xmax, z_xmax, fval)
                if fval < fmin:
                    fmin = fval
                    rec_x_xmax = x_xmax
                    rec_y_xmax = y_xmax
                    rec_z_xmax = z_xmax
    rec_r_xmax = np.sqrt(
        (rec_x_xmax - pri_xcore) ** 2
        + (rec_y_xmax - pri_ycore) ** 2
        + (rec_z_xmax - GroundAltitude) ** 2
    )
    rec_theta = 180 - np.rad2deg(np.arccos((rec_z_xmax - GroundAltitude) / rec_r_xmax))
    rec_phi = 180 + np.rad2deg(
        np.arctan2(rec_y_xmax - pri_ycore, rec_x_xmax - pri_xcore)
    )
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_r_xmax,
        rec_x_xmax,
        rec_y_xmax,
        rec_z_xmax,
        fmin,
    )


def ADF_scan_fixInfo_freedw(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    xs = pri_x_xmax + pri_xcore
    ys = pri_y_xmax + pri_ycore
    zs = pri_z_xmax

    dwList = np.linspace(0.01, 3, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    fmin = np.inf
    for dw0 in dwList:
        for A0 in Alist:
            # for B0 in Blist:
            if True:
                B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, xs, ys, zs, pri_theta, pri_phi, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_fixInfo_fixdw(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    a_par = pars[2]
    b_par = pars[3]
    c_par = pars[4]
    d_par = pars[5]
    e_par = pars[6]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    xs = pri_x_xmax + pri_xcore
    ys = pri_y_xmax + pri_ycore
    zs = pri_z_xmax

    dwList = np.linspace(0.01, 3, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    Xmax = float(pri_r_xmax / 1e4)
    dw0 = a_par / (b_par * Xmax**2 + c_par * Xmax + d_par) + e_par
    # dw0 = a + b * (180-pri_theta) + c * (180-pri_theta)**2

    fmin = np.inf
    # for dw0 in dwList:
    if True:
        for A0 in Alist:
            for B0 in Blist:
            # if True:
                # B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, xs, ys, zs, pri_theta, pri_phi, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_fixXmax_freedw(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    theta0 = pars[2]
    phi0 = pars[3]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    xs = pri_x_xmax + pri_xcore
    ys = pri_y_xmax + pri_ycore
    zs = pri_z_xmax

    dwList = np.linspace(0.01, 3, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    fmin = np.inf
    for dw0 in dwList:
        for A0 in Alist:
            # for B0 in Blist:
            if True:
                B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_fixXmax_fixdw(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    theta0 = pars[2]
    phi0 = pars[3]
    r0 = pars[4]
    x0 = pars[5]
    y0 = pars[6]
    z0 = pars[7]
    a_par = pars[8]
    b_par = pars[9]
    c_par = pars[10]
    d_par = pars[11]
    e_par = pars[12]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    # xs = pri_x_xmax + pri_xcore
    # ys = pri_y_xmax + pri_ycore
    # zs = pri_z_xmax

    dwList = np.linspace(0.01, 10, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    Xmax = float(r0 / 1e4)
    dw0 = a_par / (b_par * Xmax**2 + c_par * Xmax + d_par) + e_par
    # dw0 = a + b * (180-pri_theta) + c * (180-pri_theta)**2

    fmin = np.inf
    # for dw0 in dwList:
    if True:
        for A0 in Alist:
            for B0 in Blist:
            # if True:
                # B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, x0, y0, z0, theta0, phi0, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_freedw(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    theta0 = pars[2]
    phi0 = pars[3]
    x0 = pars[4]
    y0 = pars[5]
    z0 = pars[6]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    dwList = np.linspace(0.01, 5, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    fmin = np.inf
    for dw0 in dwList:
        for A0 in Alist:
            # for B0 in Blist:
            if True:
                B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, x0, y0, z0, theta0, phi0, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_fixdw(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    theta0 = pars[2]
    phi0 = pars[3]
    x0 = pars[4]
    y0 = pars[5]
    z0 = pars[6]
    a = pars[7]
    b = pars[8]
    c = pars[9]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    dwList = np.linspace(0.01, 5, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    dw0 = a + b * (180-pri_theta) + c * (180-pri_theta)**2

    fmin = np.inf
    if True:
    # for dw0 in dwList:
        for A0 in Alist:
            # for B0 in Blist:
            if True:
                B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, x0, y0, z0, theta0, phi0, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_fixdw_recZ(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    theta0 = pars[2]
    phi0 = pars[3]
    x0 = pars[4]
    y0 = pars[5]
    z0 = pars[6]
    a = pars[7]
    b = pars[8]
    c = pars[9]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    dwList = np.linspace(0.01, 3, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    dw0 = a + b * (180-theta0) + c * (180-theta0)**2

    fmin = np.inf
    if True:
    # for dw0 in dwList:
        for A0 in Alist:
            # for B0 in Blist:
            if True:
                B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, x0, y0, z0, theta0, phi0, dw0, A0, B0
                )
                # print(theta, phi, dw, A, B, fval)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


def ADF_scan_withNoise(pars):
    shower_parameter_file = pars[0]
    trigger_file = pars[1]
    theta0 = pars[2]
    phi0 = pars[3]
    x0 = pars[4]
    y0 = pars[5]
    z0 = pars[6]

    data = np.loadtxt(trigger_file)
    # print(data.shape)
    antid = data[:, 0]
    evtid = data[0, 1]
    ant_t = data[:, 2]
    ant_E = data[:, 3]
    ant_p = data[:, 4]

    np.random.seed(0)
    dp = np.random.normal(scale=0.1, size=len(ant_t))
    ant_p = ant_p * (1 + dp)

    evtid = int(trigger_file.split("/")[-1].split("_")[0][3:])

    shower_parameter = np.loadtxt(
        shower_parameter_file, usecols=[1, 2, 3, 5, 6, 7, 8, 0]
    )
    evtids = shower_parameter[:, -1]
    evtid = evtids == evtid

    pri_energy = shower_parameter[evtid, 2]
    pri_theta = shower_parameter[evtid, 1]
    pri_phi = shower_parameter[evtid, 0]
    pri_r_xmax = shower_parameter[evtid, 3]
    pri_x_xmax = shower_parameter[evtid, 4]
    pri_y_xmax = shower_parameter[evtid, 5]
    pri_z_xmax = shower_parameter[evtid, 6]

    pri_xcore = float(trigger_file.split("_")[-2][5:])
    pri_ycore = float(trigger_file.split("_")[-1][5:-4])

    antid = antid.astype("int")
    x = x_layout[antid]
    y = y_layout[antid]
    z = np.array(z_layout)[antid]

    dwList = np.linspace(0.01, 3, 5)
    Alist = np.linspace(5, 10, 5)
    Blist = np.linspace(-6, 0, 5)

    fmin = np.inf
    # for dw0 in dwList:
    if True:
        dw0 = 9.31 - 0.16 * pri_theta + 0.001 * pri_theta**2
        for A0 in Alist:
            # for B0 in Blist:
            if True:
                B0 = np.log10(0.005)
                theta, phi, dw, A, B, fval = ADF_fit(
                    x, y, z, ant_t, ant_p, x0, y0, z0, theta0, phi0, dw0, A0, B0
                )
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A
                    rec_B = B
    return (
        pri_energy,
        pri_theta,
        pri_phi,
        pri_r_xmax,
        pri_x_xmax,
        pri_y_xmax,
        pri_z_xmax,
        pri_xcore,
        pri_ycore,
        rec_theta,
        rec_phi,
        rec_dw,
        rec_A,
        rec_B,
        fmin,
    )


