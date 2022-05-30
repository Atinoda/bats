#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def bats(t, v, cthresh, c0_filter=None, c1_filter=None, plot=False, num=1):
    '''
    Apply the BATS algorithm to the time series observations described by t,v.

    Parameters
    ----------
    t : array-like (int or float)
        Indices for the time series observations.
    v : array-like
        Univariate observation value for each index.
    cthresh : float
        Decision threshold for point classification: C1 >= cthresh; else C0.
    c0_filter : (int or float), optional
        Temporal filter to apply to Class 0, duration is in the same units as
        the time series index. The default is None.
    c1_filter : int or float, optional
        Temporal filter to apply to Class 1, duration is in the same units as
        the time series index. The default is None.
    plot : bool, optional
        Show a diagnostic plot of algorithm operations. The default is False.
    num : int, optional
        Figure number to use for diagnostic plot. The default is 1.

    Returns
    -------
    hinges : array [h0, h1, segment class] (int)
        int array indicating segment hinges and segment class.
    seg_values : array
        float array of segment value where C1 segments are evaluated using the
        mean of their observations weighted by the time delta between indices,
        and C0 segments are shortcut evaluated to zero.

    '''

    # Diagostic plot: instantiation
    if plot:
        step_pos = 'post'
        fng = num
        fig = plt.figure(fng)
        fig.clf()
        plot_shape = (2,6)
        fig, axs = plt.subplots(nrows=plot_shape[0], ncols=plot_shape[1], sharex=True, num=fng)
        ax_thresh = plt.subplot2grid(plot_shape, (0,0), colspan=1, fig=fig)
        ax_raw = plt.subplot2grid(plot_shape, (0,1), colspan=3, fig=fig)
        ax_c0d = plt.subplot2grid(plot_shape, (0,4), colspan=1, fig=fig)
        ax_c1d = plt.subplot2grid(plot_shape, (0,5), colspan=1, fig=fig)
        ax_hist = plt.subplot2grid(plot_shape, (1,0), colspan=1, fig=fig)
        ax_seg = plt.subplot2grid(plot_shape, (1,1), colspan=3, fig=fig)
        ax_c0d_flt = plt.subplot2grid(plot_shape, (1,4), colspan=1, fig=fig)
        ax_c1d_flt = plt.subplot2grid(plot_shape, (1,5), colspan=1, fig=fig)

    vc = np.zeros(len(v))
    vnan = np.isnan(v)
    v[vnan] = -1 * np.inf
    vc[vnan] = -1
    if isinstance(cthresh, list):
        for i, cthr in enumerate(cthresh):
            vc[v >= cthr] = i + 1
    else:
        vc[v >= cthresh] = 1
    # Process timeouts
    dt = np.diff(t)
    # ti_timeout = np.argwhere(dt > timeout)
    # vc[ti_timeout] = -1
    vcd = np.diff(vc)
    hh = np.argwhere(vcd != 0) + 1
    hh = np.append([0], hh)
    ss = vc[hh]
    vv = np.ones(hh.shape)
    vv[ss==-1] = -1

    # Diagostic plot: unfiltered duration
    if plot:
        hht = t[hh].squeeze()
        hht = np.append(hht, t[-1])
        hhtd = np.diff(hht)
        cdur = hhtd
        ax = ax_c0d
        ax.hist(cdur[ss==0], bins='auto')
        ax.set_title("Class 0 Duration (unfiltered)")
        ax.yaxis.set_visible(False)
        ax = ax_c1d
        ax.hist(cdur[ss==1], bins='auto')
        ax.set_title("Class 1 Duration (unfiltered)")
        ax.yaxis.set_visible(False)

    # Apply filters
    seg_filters = []
    if c0_filter is not None:
        seg_filters.append([0,c0_filter])
    if c1_filter is not None:
        seg_filters.append([1,c1_filter])
    if len(seg_filters):
        for si, sf in seg_filters:
            # Calculate durations
            hht = t[hh].squeeze()
            hht = np.append(hht, t[-1])
            hhtd = np.diff(hht)
            # Mark invalid
            inval_mask = np.argwhere( (ss == si) & (hhtd < sf) )
            # if action == 'hold':
            hh = np.delete(hh, inval_mask)
            ss = np.delete(ss, inval_mask)
            vv = np.delete(vv, inval_mask)
            # Merge adjacent of same type
            ssd = np.diff(ss)
            ssd = (ssd != 0)
            ssd = np.append([True], ssd)
            hh = hh[ssd]
            ss = ss[ssd]
            vv = vv[ssd]
    # Produce hinge lists
    hh = hh.astype(int)
    h0 = hh
    h1 = hh[1:]
    h1 = np.append(h1, len(t)-1)
    # Find mean of segments
    idc_class1 = np.argwhere(ss == 1).squeeze()
    sv = np.zeros(len(hh))
    for ss1idx in idc_class1:
        h0i = h0[ss1idx]
        h1i = h1[ss1idx]
        sv[ss1idx] = np.average(v[h0i:h1i], weights=dt[h0i:h1i])
    # Assemble into array: h0, h1, seg-state, seg-value
    v[vnan] = np.nan  # Leave the values as they were found
    hinges = np.array([h0, h1, ss], int).transpose()
    seg_values = sv

    # Diagostic plot: final result
    if plot:
        ax = ax_thresh
        ax.set_title("Class Threshold")
        ax.hist(v, bins='auto')
        ax.axvline(cthresh, linestyle='-.', linewidth=2, color='k')
        ax.yaxis.set_visible(False)

        ax = ax_raw
        ax.step(t, v, where=step_pos, linewidth=2)
        ax.axhline(cthresh, linestyle='-.', linewidth=2, color='k')
        ax.set_title('Input Data')

        ax = ax_hist
        ax.set_title("Distributions")
        ax.hist(v, bins='auto', alpha=0.5, label='Raw')
        axtw = ax.twinx()
        axtw.hist(seg_values, bins='auto', color='r', alpha=0.5,  label='Seg')
        ax.legend(loc='upper left')
        axtw.legend(loc='upper right')
        ax.yaxis.set_visible(False)
        axtw.yaxis.set_visible(False)

        ax = ax_seg
        ax.step(t, v, where=step_pos)
        ax.fill_between(t, v, step=step_pos, alpha=0.5)
        ax.step(t[hinges[:,0]], seg_values,
                where=step_pos, color='r', linewidth=2)
        for h0i,h1i in hinges[:,:2]:
            ax.axvline(t[h0i], color='r', linestyle=':')
        ax.set_title('Segmented Data')

        cdur = hinges[:,1] - hinges[:,0]
        ax = ax_c0d_flt
        ax.set_title("Class 0 Duration (filtered)")
        ax.hist(cdur[hinges[:,2]==0], bins='auto')
        ax.yaxis.set_visible(False)
        ax = ax_c1d_flt
        ax.set_title("Class 1 Duration (filtered)")
        ax.hist(cdur[hinges[:,2]==1], bins='auto')
        ax.yaxis.set_visible(False)

        ax_c0d.get_shared_x_axes().join(ax_c0d, ax_c0d_flt)
        ax_c1d.get_shared_x_axes().join(ax_c1d, ax_c1d_flt)
        ax_raw.get_shared_x_axes().join(ax_raw, ax_seg)

        fig.tight_layout()

    return hinges, seg_values
