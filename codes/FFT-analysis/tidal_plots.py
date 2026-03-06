import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
from scipy import signal as ssignal
from scipy import stats as sstats
import bisect
from scipy.optimize import minimize_scalar
import matplotlib.ticker as plticker
import ttide
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import matplotlib
import warnings


def run_ttide(U: np.ndarray, V: np.ndarray, xin_units: str, 
              time_lim: np.ndarray, latitude: float,
              constitnames: list):
    """
    Wrapper for ttide.t_tide()

    inputs:
    - U: east-west velocity component for a single bin, 1-D, in m/s
    - V: north-south velocity component for a single bin, 1-D, in m/s
    - xin_units: units to apply to the xin parameter for ttide.t_tide()
    - time_lim: time array
    - latitude: latitude of ADCP
    - constitnames: list of tidal constituent names for which to compute the frequency, amplitude, and phase
    """
    mask_notna = ~(np.isnan(U) | np.isnan(V))
    U[mask_notna] = np.nanmean(U)
    V[mask_notna] = np.nanmean(V)


    xin = U + 1j * V
    if xin_units == 'cm/s':
        xin *= 100

    # Round to 3 decimal places, units are hours
    sampling_interval = np.round(pd.Timedelta(time_lim[1] - time_lim[0]).seconds / 3600, 3)
    ray = 1
    synth = 0
    stime = pd.to_datetime(time_lim[0])

    result = ttide.t_tide(
        xin=xin,
        dt=sampling_interval,  # Sampling interval in hours, default = 1
        stime=stime,  # The start time of the series
        lat=latitude,
        constitnames=constitnames,  # ttide_constituents(datetime_lim, sampling_interval, ray),
        shallownames=[],
        synth=synth,  # The signal-to-noise ratio of constituents to use for the "predicted" tide
        ray=ray,  # Rayleigh criteria, default 1
        lsq='direct',  # avoid running as a long series
        out_style=None  # No printed output
    )
    return result


def find_constituent(con_names, constituent):
    """ Find the index of the requested constituent.
    Author: Maxim Krassovski"""

    # Check that the list of names is in unicode and not byte.
    tmp = con_names.astype('U4')
    ind = np.where(tmp == constituent)[0]
    if len(ind) == 0:
        return None
    else:
        return ind[0]


def parse_tidal_constituents(constituents, tide_result):
    """ Given the full tidal output from ttide, return the specified constituents.
    Author: Maxim Krassovski

    Parameters
    ----------
    constituents : list
        Constituents to parse
    tide_result : dict
        ttide output

    Returns
    -------
    dictionary of tuples,
    with the keys being the constituent names and the tuples being either (amp,phase) or (maj,min,inc,phase) if complex.
    """

    tidal_par, error_par = {}, {}

    for con in constituents:
        if 'nameu' in tide_result.keys():
            ind = find_constituent(tide_result['nameu'], con.ljust(4))
        else:
            ind = None  # case where tidal analysis was not performed
        # todo: option for obs but not mod and vice versa?
        if ind is not None:
            # Quantity, error
            a, ea = tide_result['tidecon'][ind, 0], tide_result['tidecon'][ind, 1]
            b, eb = tide_result['tidecon'][ind, 2], tide_result['tidecon'][ind, 3]
            if tide_result['tidecon'].shape[1] == 8:
                # Has complex components
                c, ec = tide_result['tidecon'][ind, 4], tide_result['tidecon'][ind, 5]
                d, ed = tide_result['tidecon'][ind, 6], tide_result['tidecon'][ind, 7]
            else:
                c, ec = np.nan, np.nan
                d, ed = np.nan, np.nan

        else:
            a, ea = np.nan, np.nan
            b, eb = np.nan, np.nan
            c, ec = np.nan, np.nan
            d, ed = np.nan, np.nan

        tidal_par[con] = (float(a), float(b), float(c), float(d))
        error_par[con] = (float(ea), float(eb), float(ec), float(ed))
    return tidal_par, error_par


# this is a duplicate of the subroutine in pkg_plot_currents.py
def calculate_ellipse(major: float, minor: float, inc: float, phase: float):
    """
    Reads in the tidal parameters, returns a series of dots for plotting.
    (Easier than trying to describe the ellipse analytically)
    May need to be reworked for plotting ellipses on a map.
    From
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_plot_currents_comparison.py?ref_type=heads

    major, minor, inc, phase: parameters returned from ttide.t_tide() in the 'tidecon' key value
    """
    # major, minor, inc, phase = tidal_params
    # todo: add error checking here

    # Convert inclination and phase to radians
    inc = inc * np.pi / 180.0
    phase = phase * np.pi / 180.0

    # Eccentricity
    okmajor = np.nonzero(major)
    if len(okmajor[0]) > 0:
        ecc = minor / major
    else:
        ecc = np.nan
        major = np.nan

    # Ellipse
    nsegs = 3600
    t = np.linspace(0, 2.0 * np.pi, nsegs)
    ellipse = major * (np.cos(t[:] - phase) + 1j * ecc * np.sin(t[:] - phase)) * np.exp(1j * inc)

    return ellipse


def plot_ellipse(ellipse, gap, clr):
    """Takes in a calculated ellipses and plots it, with a dot indicating a particle at the end of
    tracing out the ellipse.
    From
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_plot_currents_comparison.py?ref_type=heads
    """
    # Plot real and imaginary components of the ellipse
    plt.plot(np.real(ellipse[10 * gap:3600]), np.imag(ellipse[10 * gap:3600]), color=clr, linewidth=1)
    plt.scatter(np.real(ellipse[-1]), np.imag(ellipse[-1]), s=3, c=clr, marker='.')
    return


def best_tick(span, mostticks):
    """
    Author : Maxim Krassovski
    """
    # https://stackoverflow.com/questions/361681/algorithm-for-nice-grid-line-intervals-on-a-graph
    minimum = span / mostticks
    magnitude = 10 ** math.floor(math.log(minimum, 10))
    residual = minimum / magnitude
    # this table must begin with 1 and end with 10
    table = [1, 2, 5, 10]  # options for gradations in each decimal interval
    tick = table[bisect.bisect_right(table, residual)] if residual < 10 else 10
    return tick * magnitude


def make_plot_tidal_ellipses(
        station: str,
        instrument_depth, latitude: float,
        time_lim: np.ndarray, bin_depth_lim: np.ndarray, 
        ns_lim: np.ndarray,
        ew_lim: np.ndarray
):
    """
    Follow usage in
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_tides.py?ref_type=heads#L148

    Plot vertical distribution of tidal ellipses

    From ttide documentation:
    Although missing data can be handled with NaN, it is wise not to
    have too many of them. If your time series has a lot of missing
    data at the beginning and/or end, then truncate the input time
    series.
    """

    def plot_tidal_ellipse(ellipse, el_color, xoff=0., yoff=0., scale=1):
        # plot conjugated ellipse for inverted y-axis
        gap = 0
        plot_ellipse(
            np.conjugate(ellipse) * scale + (xoff + 1j * yoff), 
            gap, 
            el_color
        )

    def ellipses_lims(scale):
        """ Data limits for scaled and shifted ellipses """
        x_span, y_span, margin, y_data_lim = ellipses_span(scale)
        x_data_lim = -(x_span / 2 + margin), x_span / 2 + margin
        y_data_lim = y_data_lim[0] - np.sign(y_span) * margin, y_data_lim[1] + np.sign(y_span) * margin
        return x_data_lim, y_data_lim

    def aspect_func(scale):
        """ Difference between aspect of the area occupied by ellipses and axes aspect.
        Used in ellipse scale optimization.
        """
        x_span, y_span, margin, _ = ellipses_span(scale)
        return abs(
            ax_aspect - (abs(y_span) + margin) / (x_span + margin)
        )

    def ellipses_span(scale):
        """ Span of scaled and shifted ellipses along x- and y-axis and including y_min, y_max """
        els = np.array([])
        # for el in ell_dict.keys():
        for dep in bin_dict.keys():
            for tc in bin_dict[dep]['ellipses'].keys():
                try:
                    els = np.append(
                        els, 
                        bin_dict[dep]['ellipses'][tc] * scale + (0 + 1j * dep)
                    )
                    # els = np.append(els, el[1] * scale + (0 + 1j * el[0]))  # el[0] is depth, el[1] is ellipse points
                except:
                    continue
                    # els = np.append(els, 0+0j )
        x_span = 2 * np.nanmax(np.abs(np.real(els)))
        y_data_lim = (
            min([y_min, np.nanmin(np.imag(els))]), 
            max([y_max, np.nanmax(np.imag(els))])
        )
        y_span = y_data_lim[1] - y_data_lim[0]
        margin = 0.1 * min(x_span, abs(y_span))
        return x_span, y_span, margin, y_data_lim

    def set_lims(a):
        """ Ensure axes limits include x_lim, y_lim. Works only for non-inverted axes. """
        if np.min(np.abs(a.get_xlim())) < x_lim[1]:
            a.set_xlim(x_lim)
        a_ylim = a.get_ylim()
        if a_ylim[0] > y_lim[0] or a_ylim[1] < y_lim[1]:
            a.set_ylim(
                min(a_ylim[0], y_lim[0]), max(a_ylim[1], y_lim[1])
            )
        return

    instrument_depth = np.round(instrument_depth)

    # Major tidal constituents
    major_constit = ['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2']
    bin_dict = {}

    # Iterate through the bins
    # two-level dict with bin depth first level then tidal const. second level
    for i in range(len(ew_lim)):
        result = run_ttide(
            U=ew_lim[i, :],
            V=ns_lim[i, :],
            xin_units='m/s',
            time_lim=time_lim,
            latitude=latitude,
            constitnames=major_constit
        )

        # Find the (maj,min,inc,phase) for each tidal constituent and use them to calculate the ellipse
        tidal_par, error_par = parse_tidal_constituents(
            constituents=major_constit, tide_result=result
        )

        ell_dict = {}
        for k, name in enumerate(major_constit):
            # result['tidecon'] has columns for
            # major, major err, minor, minor err, inc (inclination), inc err, phase, phase err.
            # Missing leading column for freq stored in result['fu']
            # Missing trailing column for SNR stored in result['snr']
            ell_dict[name] = calculate_ellipse(
                major=tidal_par[name][0],  # result['tidecon'][k, 0],
                minor=tidal_par[name][1],  # result['tidecon'][k, 2],
                inc=tidal_par[name][2],  # result['tidecon'][k, 4],
                phase=tidal_par[name][3],  # result['tidecon'][k, 6]
            )

        # Add to dictionary containing all the results
        bin_dict[bin_depth_lim[i]] = {
            'result': result, 'ellipses': ell_dict
        }

    # # ellipse points for plotting
    # ell_list = [[(k, calculate_ellipse(el_par_z)) for k, el_par_z in zip(ozk, el_par_case)]
    #             for el_par_case in ell_params]
    # # make a flat list for all cases for unified scaling calculation
    # combined_ell_list = []
    # for one in ell_list:
    #     combined_ell_list.extend(one)

    # include these limits in the plot even if ellipses don't span to them
    buffer = 10  # meters
    y_min = 0
    y_max = np.max(bin_depth_lim) + buffer  # ozk.max()

    # Make the plot
    n_cases = len(major_constit)  # Number of tidal constituents??
    fig, axes = plt.subplots(
        1, n_cases, sharey='row', figsize=(1.5 * n_cases, 8)
    )
    plt.subplots_adjust(wspace=.0)
    title = f'{station}-{deployment_number} tidal ellipses'
    fig.suptitle(title)

    for ax in axes:
        ax.set_aspect('equal', adjustable='datalim')  # this is for correct ellipse aspect
        ax.set_xlabel('Ellipse Size\n(cm/s)')  # do before plt.tight_layout() otherwise gets cut off

    axes[0].set_ylabel('Depth (m)')  # do before plt.tight_layout() otherwise gets cut off

    # do tight_layout before axes[0].get_ylim() for correct numbers (it renders fig?)
    plt.tight_layout()
    # axes aspect to fit the ellipses in
    ax_aspect = abs(np.diff(axes[0].get_ylim()) / np.diff(axes[0].get_xlim()))
    # best ellipse scale to fit the axes
    with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
        optim = minimize_scalar(aspect_func)
    ell_scale = optim.x[0]
    x_lim, y_lim = ellipses_lims(ell_scale)
    # plot ellipses
    for ax, ell_case in zip(axes, major_constit):
        plt.sca(ax)
        for depth in bin_dict.keys():
            # el[0] is depth, el[1] is ellipse points
            plot_tidal_ellipse(
                bin_dict[depth]['ellipses'][ell_case], 
                'k', 
                yoff=depth, 
                scale=ell_scale
            )
        set_lims(ax)  # make sure ellipses are not clipped

    # same xtick for all axes
    tick_base = best_tick(np.ptp(axes[0].get_xlim()) / ell_scale, 6)
    loc = plticker.MultipleLocator(base=tick_base * ell_scale)

    # scaled x-ticks, labels and annotations
    for ax, case in zip(axes, major_constit):
        ax.xaxis.set_major_locator(loc)
        x_ticks = plticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x / ell_scale)
        )
        ax.xaxis.set_major_formatter(x_ticks)
        # ax.set_xlabel('Ellipse Size (m/s)')  # moved to earlier in the function
        ax.grid(which='both')
        ax.text(
            0.05, 0.98, case, ha='left', va='top', 
            transform=ax.transAxes
        )

    axes[0].invert_yaxis()  # do it only once; shared y-axes applies it to the rest
    # axes[0].set_ylabel('Depth (m)')

    # plt.show()

    # Save the figure
    # plot_name = f'{station}_{instrument_depth}m_tidal_ellipses.png'

    # _, plot_name = review_plot_naming(
    #     plot_title=None, png_name=plot_name, serial_number=serial_number, is_pre_split=False, resampled=resampled
    # )

    # Create L1_Python_plots or L2_Python_plots subfolder if not made already
    # plot_dir = get_plot_dir(data_filename, dest_dir)
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)

    # plot_name = os.path.join(plot_dir, plot_name)
    # plt.savefig(plot_name)
    # plt.close()
    return # plot_name


def sampling_freq(time) -> float:
    """
    Return fs in units of CPD (cycles per day)
    """
    s_per_day = 3600 * 24
    dt_s = (time[1] - time[0]).astype('timedelta64[s]')  # seconds
    return np.round(s_per_day / dt_s.astype(np.float32))


# noinspection GrazieInspection
def rot(u, v=None, fs=1.0, nperseg=None, noverlap=None, 
        detrend='constant', axis=-1, conf=0.95):
    """ Rotary and cross-spectra with coherence estimate.
    Function from https://gitlab.com/krassovski/pyap/-/raw/master/analysispkg/pkg_spectrum.py
    Parameters
    ----------
    u : array_like
        Sampled series, e.g. u-component of velocity.
    v : array_like, optional
        Second series or v-component of velocity.
    fs : float, optional
        Sampling frequency of the series. Defaults to 1.0.
    nperseg : int, optional
        Length of spectral window. Defaults to the power of 2 nearest to the
        quarter of the series length.
    noverlap : int, optional
        Number of points to overlap between windows. If `None`,
        ``noverlap = nperseg / 2``. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).
    conf : float, optional
        Confidence limits for power spectra and confidence level for coherence to calculate,
        e.g. conf=0.95 gives 95% confidence limits. Defaults to 0.95.

    Returns
    -------
    Dictionary with fields:
        dof     number of degrees of freedom
        f       frequency vector
        period  corresponding period
        pxx     x-component spectrum (u)
        pyy     y-component spectrum (v)
        cxy     co-spectrum
        qxy     quad-spectrum
        hxy     admittance (transfer,response) function
        hhxy    ratio of sx and sy
        r2xy    coherence squared
        phase   phase
        pneg    clockwise component of rotary spectra
        ppos    counter-clockwise component of rotary spectra
        ptot    total spectra
        orient  orientation of major axis of the ellipse
        rotcoeff    rotary coefficient
        stabil  stability of the ellipse
        conf    upper and lower limits of the confidence interval relative to unity
        cohconf confidence level for coherence squared

    Adapted from a Fortran routine ROTCOL written by A.Rabinovich.
    Reference: Gonella, 1972, DSR, Vol.19, 833-846
    """
    n = u.shape[axis]  # series length

    if nperseg is None:
        nperseg = int(2 ** np.round(np.log2(n / 4)))
    if noverlap is None:
        noverlap = np.floor(nperseg / 2)

    k = np.fix(n / nperseg) + np.fix((n - nperseg / 2) / nperseg)

    # power per unit frequency
    f, sx = ssignal.welch(
        u, fs=fs, nperseg=nperseg, noverlap=noverlap, 
        detrend=detrend, axis=axis
    )

    sx = np.real(sx)  # needed in case there are NaNs in u (sx is NaN+i*NaN)
    if v is not None:
        _, sy = ssignal.welch(
            v, fs=fs, nperseg=nperseg, noverlap=noverlap, 
            detrend=detrend, axis=axis
        )
        sy = np.real(sy)  # needed in case there are NaNs in v  # TODO Check

        # cross-spectra
        # power per unit frequency
        _, pxy = ssignal.csd(
            u, v, fs=fs, nperseg=nperseg, noverlap=noverlap, 
            detrend=detrend, axis=axis
        )

        cxy = np.real(pxy)  # co-spectrum
        qxy = -np.imag(pxy)  # quadrature spectrum
        hxy = abs(pxy) / sx  # admittance function
        hhxy = sy / sx  # component spectra ratio

        r2xy = np.abs(pxy) ** 2 / (sx * sy)  # coherence squared
        phase = np.degrees(-np.angle(pxy))  # phase

        # rotary spectra
        sm = (sx + sy - 2 * qxy) / 2
        sp = (sx + sy + 2 * qxy) / 2
        st = sm + sp

        # major axis orientation
        orient = np.degrees(np.arctan2(cxy * 2, sx - sy) / 2)

        # rotary coefficient
        rotcoeff = (sm - sp) / st

        # stability of the ellipse
        stabil = np.sqrt(
            (st ** 2 - (sx * sy - cxy ** 2) * 4) / 
                (st ** 2 - 4 * qxy ** 2)
        )

        # confidence level for coherence
        # dof = k * 2
        # confidence = np.array([.99, .95, .90, .80])
        # cohconf = 1 - (1 - confidence) ** (2 / dof)
        cohconf = 1 - (1 - conf) ** (1 / k)
    else:
        sy = cxy = qxy = hxy = hhxy = r2xy = phase = sm = sp = st = orient = rotcoeff = stabil = cohconf = None

    # confidence intervals
    vw = 1.5  # coefficient for hann window
    # k = np.floor((n - noverlap) / (nperseg - noverlap))  # number of windows
    dof = 2 * np.round(k * 9 / 11 * vw)  # degrees of freedom
    # c = dof / np.array(scipy.stats.chi2.interval(conf, df=dof))

    # TSA way (takes 50% overlap into account and applies window coefficient):
    # conf_lims = chi2conf(confidence, dof);    % limits relative to 1
    conf_lims = dof / np.array(sstats.chi2.interval(conf, df=dof))
    # conf_lims = np.array(scipy.stats.chi2.interval(conf, df=dof))
    # confu = conf_lims[0]
    # confl = conf_lims[1]
    # confLims = sm * confLims1  # limits for each spectral value

    with np.errstate(invalid='ignore', divide='ignore'):
        period = 1 / f
    # pack all results in an output dict
    r = dict(dof=dof, f=f, period=period, pxx=sx, pyy=sy, cxy=cxy, 
             qxy=qxy, hxy=hxy, hhxy=hhxy, r2xy=r2xy, phase=phase,
             pneg=sm, ppos=sp, ptot=st, orient=orient, rotcoeff=rotcoeff,
             stabil=stabil,
             conf=conf_lims, cohconf=cohconf)
    return r


def plot_spectrum(f, p, c=None, clabel=None,
                  cx=None, cy=None,
                  color=None, ccolor='k',
                  units='m', funits='cpd',
                  ax=None, **options):
    """ Spectrum plot in log-log axes with confidence interval
    Function from https://gitlab.com/krassovski/pyap/-/raw/master/analysispkg/pkg_spectrum.py
    Parameters
    ----------
    f : array_like
        Array of sample frequencies.
    p : array_like
        Power spectral density or power spectrum of x.
    c : array of length 2, optional
        Upper and lower confidence limit relative to 1. Returned by spectrum()
        in `c`. No error bar by default.
    clabel : str, optional
        Label for the error bar, e.g. '95%'. No label by default.
    cx,cy : float, optional
        Coordinates (in data units) for the error bar. By default, cx is at
        0.8 of x-axis span, and cy is at 3 lower confidence intervals above max
        power value in the band spanning the error bar with label.
    color : matplotlib compatible color, optional
        Color cpecification for the plot line. Defaults to Matplotlib default
        color.
    ccolor : matplotlib compatible color or 'same', optional
        Color specification for the error bar. If 'same', uses same color as
        the plot line. Defaults to black, 'k'.
    units : str, optional
        Units for analysed series, e.g. 'm' for sea level, 'm/s' for velocity.
        Defaults to 'm'.
    funits : str, optional
        Frequency units. Defaults to 'cpd'.
    ax : Matplotlib axes handle, optional
        Defaults to the current axes.
    **options : kwargs
        Extra options for matplotlib's loglog()
    """

    # exclude zero-frequency value
    ival = f != 0
    f = f[ival]
    p = p[ival]

    if ax is None:
        ax = plt.gca()

    # spectrum
    hp, = ax.loglog(f, p, color=color, **options)
    ax.set_xlabel('Frequency (' + funits + ')')
    ax.set_ylabel(r'PSD (' + units + ')$^2$/' + funits)
    #    ax.grid(True,which='both')
    ax.grid(True, which='major', linewidth=1)
    ax.grid(True, which='minor', linewidth=.5)

    # error bar
    if c is not None:
        # attempt of automatic error bar placement
        if cx is None or cy is None:
            xlim = ax.get_xlim()
            xliml = np.log10(xlim)
            dxliml = np.abs(np.diff(xliml))
            if cx is None:
                cx = 10 ** (np.min(xliml) + dxliml * .8)  # at 0.8 of x-axis span
            if cy is None:
                # approx x-range for error bar
                xrng = 10 ** (np.log10(cx) + dxliml * [-0.05, 0.15])
                # find p within the error bar horizontal span, extrapolate if necessary
                inx = np.logical_and(f > xrng[0], f < xrng[1])
                if np.any(inx):
                    pxr = p[inx]
                else:
                    pxr = np.interp(xrng, f, p)
                # 3 lower intervals above max power value in the band
                cy = pxr.max() * (1 / c[1]) ** 3
        if ccolor == 'same':
            ccolor = hp[0].get_color()
        # Need an error bar that goes from cy*c[1] to cy*c[0];
        # plt.errorbar plots bar from cy-clower to cy+cupper;
        # convert c to yerr accordingly:
        yerr = cy * ((c[::-1] - 1) * [-1, 1])[:, None]
        ax.errorbar(cx, cy, yerr=yerr, color=ccolor, capsize=5, marker='o', markerfacecolor='none')
        # capsize is the width of horizontal ticks at upper and lower bounds in points
        if clabel is not None:
            ax.text(cx, cy, '   ' + clabel, verticalalignment='center', color=ccolor)
    return hp


def mark_frequency(fmark, fname='', ax=None, f=None, p=None, fig=None):
    """ Annotation arrow with text label above spectra plot at a specified frequency
    Author: Maxim Krassovski
    https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_spectrum.py?ref_type=heads

    Parameters
    ----------

    fmark : float
        Frequency to mark
    fname : str, optional
        Text for annotation. Defaults to empty string.
    ax : axes handle, optional
        Use all line objects in these axes to determine vertical position of
        the annotation. If supplied, f,p are ignored. Also denotes axes for
        plotting. Defaults to None.
    f,p : array_like, optional
        x- and y-coordinates for the curve to use for vertical positioning of
        the annotation. Defaults to None.
    fig : matplotlib figure or None
        Figure instance for which single bin rotary spectra have been plotted
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # determine y-position from all lines in the axes
    if f is not None and p is not None:  # determine y-position from supplied f,p
        pmark = np.interp(fmark, f, p)
    else:
        lines = ax.lines
        # find highest line at this freq
        pmark = np.nan
        for ln in lines:
            pmark = np.fmax(pmark,
                            np.interp(fmark, ln.get_xdata(), ln.get_ydata(),
                                      left=np.nan, right=np.nan))
        if np.isnan(pmark):  # no lines at this frequency
            # place it in the middle
            ylim = ax.get_ylim()
            if ax.get_yscale() == 'log':
                pmark = 10 ** (np.mean(np.log10(ylim)))
            else:
                pmark = np.mean(ylim)

        # raise RuntimeError('Either axes or f,p values are needed to determine y-position for frequency mark.')

    arrowprops = dict(arrowstyle="simple, head_width=0.25, tail_width=0.05",
                      color='k', lw=0.5, shrinkB=10)
    # shrinkB is the offset in pixels from (f,p) line

    ann = ax.annotate(fname, xy=(fmark, pmark), xytext=(0, 25),
                      textcoords='offset points', horizontalalignment='center',
                      arrowprops=arrowprops)

    # matplotlib.use('qt4agg')
    fig.canvas.draw()
    box = matplotlib.text.Text.get_window_extent(ann)
    ylim = ax.get_ylim()
    yann = ax.transData.inverted().transform(box)[:, 1]
    # from IPython import embed;    embed()  ######################
    if ylim[1] < yann[1]:
        ax.set_ylim([ylim[0], yann[1] * yann[1] / yann[0]])
        # ax.grid()
    return


def plot_rot(r: dict, clabel=None, cx=None, cy=None, color=None, ccolor='k', units='m/s', funits='cpd',
             fig=None, ax_neg=None, ax_pos=None, **options):
    """ Plot rotary components spectra
    Function from https://gitlab.com/krassovski/pyap/-/raw/master/analysispkg/pkg_spectrum.py
    fig : matplotlib figure or subfigure
    """
    if fig is None and (ax_pos is None and ax_neg is None):
        fig = plt.figure(figsize=(8.5, 4.5))  # (width, height) so the subplots are more square
    if ax_pos is None and ax_neg is None:
        (ax_neg, ax_pos) = fig.subplots(1, 2, sharey=True)  # , subplot_kw={'aspect': 'equal'})
        ax_neg.invert_xaxis()
    if clabel is None:
        c = None
    else:
        c = r['conf']

    hneg = plot_spectrum(r['f'], r['pneg'], color=color, ccolor=ccolor,
                         units=units, funits=funits, ax=ax_neg, **options)
    hpos = plot_spectrum(r['f'], r['ppos'], c=c, clabel=clabel, cx=cx, cy=cy, color=color, ccolor=ccolor,
                         units=units, funits=funits, ax=ax_pos, **options)
    ax_pos.set(ylabel=None)
    # fig.tight_layout()
    return fig, ax_neg, ax_pos, hneg, hpos


def make_plot_rotary_spectra(
        station: str, 
        instrument_depth: float, 
        bin_number: int, 
        bin_depths_lim: np.ndarray, 
        time_lim: np.ndarray,
        ns_lim: np.ndarray, ew_lim: np.ndarray, latitude: float,
        axis=-1, do_tidal_annotation=True
):
    """
    Make single-bin plot of rotary spectra with separate subplots for CW and CCW components

    from https://gitlab.com/krassovski/pyap/-/blob/master/analysispkg/pkg_spectrum.py?ref_type=heads
    Follow functions rot() and plot_rot()

    fs: sampling frequency in units of CPD

    Notes:
    NaNs need to be treated before applying this function. Maxim recommends to fill with mean or linearly interpolate
    if the tide is strong a better way would be to remove tide, fill gaps as above and add the tide back in
    long gaps will distort the spectra, so avoid showing spectra for very gappy series if possible, or specify the
    amount of missing data as a disclaimer

    inputs
    - bin_number: bin index, starts at zero, applies to subsetted data not the complete dataset at this point...

    """

    instrument_depth = np.round(instrument_depth)

    # Cycles per hour to cycles per day
    cph_to_cpd = 24

    fs = sampling_freq(time_lim)

    # Remove/replace nans from velocity components otherwise they propagate in ssignal.welch()
    u = ew_lim[bin_number, :]
    u[np.isnan(u)] = np.nanmean(u)
    v = ns_lim[bin_number, :]
    v[np.isnan(v)] = np.nanmean(v)

    bin_depth = bin_depths_lim[bin_number]

    r = rot(u=u, v=v, axis=axis, fs=fs)

    fig, ax_neg, ax_pos, hneg, hpos = plot_rot(r)

    # Add annotation of major frequencies M2 and K1 to ax_neg and ax_pos
    fnames = ['M2', 'K1']

    if do_tidal_annotation:
        # keep or filling of nans does not affect result frequencies
        try:
            result = run_ttide(
                U=u,
                V=v,
                xin_units='m/s',
                time_lim=time_lim,
                latitude=latitude,
                constitnames=fnames
            )
        except ValueError as e:
            print('t_tide failed with error', e)
            return

        # nameu might be in different order than fnames !!
        for fname_S4, fmark_cph in zip(result['nameu'], result['fu']):
            # Convert component name to string and remove any trailing whitespace
            fname = fname_S4.astype('str').strip(' ')
            # Convert frequency fmark from per hour to per day
            fmark_cpd = fmark_cph * cph_to_cpd
            mark_frequency(
                fmark=fmark_cpd, fname=fname, ax=ax_neg, f=r['f'], 
                p=r['pneg'], fig=fig
            )
            mark_frequency(
                fmark=fmark_cpd, fname=fname, ax=ax_pos, f=r['f'], 
                p=r['ppos'], fig=fig
            )

    # Add titles to the subplots
    ax_neg.set_title('CW')
    ax_pos.set_title('CCW')

    plt.suptitle(
        f'{station} Rotary Spectra - {np.round(bin_depth, 2)}m bin'
    )

    # plt.show()

    return 


def rot_freqinterp(rot_dict: dict) -> tuple:
    # Collect all frequency values from all spectra and make this the target frequency vector
    ftarget = np.array([])
    for depth in rot_dict.keys():
        fnew = rot_dict[depth]['f']
        ftarget = np.unique(np.concatenate((ftarget, fnew)))

    # Interpolate spectra values to "standard" frequencies

    # Initialize 2d arrays with shape (depth, num_frequencies)
    pneg_interp = np.zeros(len(rot_dict) * len(ftarget)).reshape((len(rot_dict), len(ftarget)))
    ppos_interp = np.zeros(len(rot_dict) * len(ftarget)).reshape((len(rot_dict), len(ftarget)))

    # Do 1d interpolation
    for i, depth in enumerate(rot_dict.keys()):
        # Default interpolation type is linear
        func_pneg = interp1d(x=rot_dict[depth]['f'], y=rot_dict[depth]['pneg'])
        func_ppos = interp1d(x=rot_dict[depth]['f'], y=rot_dict[depth]['ppos'])
        # Apply the returned functions
        pneg_interp[i, :] = func_pneg(ftarget)
        ppos_interp[i, :] = func_ppos(ftarget)

    return ftarget, pneg_interp, ppos_interp


def pcolor_rot_component(station: str,
                         instrument_depth, 
                         x: np.ndarray, y, c: np.ndarray, 
                         clim: tuple, neg_or_pos: str,
                         funits='cpd', resampled=False):
    """
    x: depth
    y: standard frequencies; the product of interpolation
    c: real component of pneg_interp or ppos_interp
    clim: c limits in format (min, max)
    neg_or_pos: "neg" for negative (CW) component, "pos" for positive (CCW) component
    """
    # % CW: R, z, componentfield, separateaxes, units
    # [~,climcw,hcw] = pcolor_component(R,z,'sm',separateaxes,units); % ,'neg',[prefix '_cw']
    # % CCW
    # [~,climccw,hccw] = pcolor_component(R,z,'sp',separateaxes,units); % ,'pos',[prefix '_ccw']
    fig, ax = plt.subplots()
    f1 = ax.pcolormesh(x, y, c, cmap='jet', shading='auto',
                       norm=LogNorm(vmin=clim[0], vmax=clim[1]))  # Maxim's code uses the jet colormap

    # Make x axis log scale
    ax.set_xscale('log')

    # Invert y axis
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymax, ymin))

    # Add legend
    cbar = fig.colorbar(f1, ax=ax)
    cbar.set_label(r'PSD (m/s)$^2$/' + funits)

    ax.set_xlabel(f'Frequency ({funits})')
    ax.set_ylabel('Depth (m)')

    title = f'{station} Rotary Spectra (XX)'
    if neg_or_pos == 'neg':
        title = title.replace('XX', 'neg./CW')
    else:
        title = title.replace('XX', 'pos./CCW')
    ax.set_title(title)

    plt.tight_layout()

    # plt.show()
    return 


def make_depth_prof_rot_spec(
        station: str,
        instrument_depth,
        bin_depths_lim: np.ndarray, ns_lim: np.ndarray,
        ew_lim: np.ndarray, time_lim: np.ndarray
):
    """
    Plot depth profiles of rotary spectra in a pseudo-color (pcolor) plot

    Adapted from Maxim Krassovski's MatLab code
    Calculate rotary spectra: https://gitlab.com/krassovski/Tools/-/blob/master/Signal/series/adcp_rot.m?ref_type=heads
    Plot them: https://gitlab.com/krassovski/Tools/-/blob/master/Signal/series/adcp_report.m?ref_type=heads
    -> rot_pcolor_plot()
    """
    instrument_depth = np.round(instrument_depth)

    fs = sampling_freq(time_lim)

    # Remove nans
    u = ew_lim
    u[np.isnan(u)] = np.nanmean(u)
    v = ns_lim
    v[np.isnan(v)] = np.nanmean(v)

    # Initialize dict to hold results for each bin
    rot_dict = {}

    # Iterate through all the bins
    for bin_idx in range(len(bin_depths_lim)):
        rot_dict[bin_depths_lim[bin_idx]] = rot(
            u=u[bin_idx, :], v=v[bin_idx, :], axis=0, fs=fs
        )

    # Standard frequencies
    ftarget, pneg_interp, ppos_interp = rot_freqinterp(rot_dict)

    # Exclude zero frequency, assuming zero frequency is first element in ftarget
    ftarget = ftarget[1:]
    pneg_interp = pneg_interp[:, 1:]
    ppos_interp = ppos_interp[:, 1:]

    # Get color range to unify both negative and positive plots
    cneg = np.real(pneg_interp)
    cpos = np.real(ppos_interp)
    cmin = np.min([np.min(cneg), np.min(cpos)])
    cmax = np.max([np.max(cneg), np.max(cpos)])

    if cmin >= cmax:
        clim = None
    else:
        clim = (cmin, cmax)

    # pcolor plot, skipping the 0 frequency
    pcolor_rot_component(
        station,
        instrument_depth,
        x=ftarget,
        y=rot_dict.keys(),
        c=cneg,
        clim=clim,
        neg_or_pos='neg',
    )
    # Positive
    pcolor_rot_component(
        station,
        instrument_depth,
        x=ftarget,
        y=rot_dict.keys(),
        c=cpos,
        clim=clim,
        neg_or_pos='pos',
    )

    # Spectra for selected bins/depths?

    return 
