#! /usr/bin/env python
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

from matplotlib.pylab import Line2D
from matplotlib import rc, font_manager



def plot_bridge_data(start_times, dv_v, load):
    rgba0 = (255/255,255/255,255/255,1)
    rgba1 = (216/255,27/255,96/255,1)
    rgba2 = (30/255,136/255,229/255,1)
    rgba3 = (255/255,183/255,7/255,1)
    rgba4 = (0/255,77/255,64/255,1)
    rgba5 = (0/255,0/255,0/255,1)
    load_color = []
    for i in range(len(load)):
        if float(load[i]) == 900.0:
            load_color.append(rgba4)
        if float(load[i]) == 600.0:
            load_color.append(rgba3)
        if float(load[i]) == 300.0:
            load_color.append(rgba2)
        if float(load[i]) == 0.0:
            load_color.append(rgba1)

    Sstart_times = []
    for i in range(len(start_times)):
        Sstart_times.append(start_times[i].datetime)

    stress_changes0 = [
                        dt.datetime(2021, 10, 7, 9, 8, 0),   # 450 -> 400
                        dt.datetime(2021, 10, 7, 10, 25, 0), # 400 -> 350
                        dt.datetime(2021, 10, 7, 11, 33, 0), # 350 -> 300
                        dt.datetime(2021, 10, 7, 12, 38, 0), # 300 -> 250
                        dt.datetime(2021, 10, 7, 13, 49, 0), # 250 -> 200
                    ]
    stress_changes1 = [
                        dt.datetime(2021, 10, 8, 7, 15, 0), # 200 -> 250
                        dt.datetime(2021, 10, 8, 7, 35, 0), # 250 -> 300
                        dt.datetime(2021, 10, 8, 7, 57, 0), # 300 -> 350
                        dt.datetime(2021, 10, 8, 8, 20, 0), # 350 -> 400
                        dt.datetime(2021, 10, 8, 9, 31, 0), # 400 -> 450
                    ]
   
    stress_colors = [
                    (100/255,100/255,100/255,1),
                    (128/255,128/255,128/255,1),
                    (182/255,182/255,182/255,1),
                    (210/255,210/255,210/255,1),
                    (225/255,225/255,225/255,1),
                    (250/255,250/255,250/255,1),
                    ]

    sizeOfFont = 14
    fontProperties = {'weight' : 'normal', 'size' : sizeOfFont}
    rc('font',**fontProperties)

    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(12, 7))


    for i in range(len(Sstart_times)):
        line2 = ax0.plot(Sstart_times[i], dv_v[i], marker='o', color=load_color[i], markersize=4, alpha=1)
        line2 = ax1.plot(Sstart_times[i], dv_v[i], marker='o', color=load_color[i], markersize=4, alpha=1)

    ax0.axvspan(Sstart_times[0]-dt.timedelta(seconds=120), stress_changes0[0], color=stress_colors[0])
    ax0.axvspan(stress_changes0[0], stress_changes0[1], color=stress_colors[1])
    ax0.axvspan(stress_changes0[1], stress_changes0[2], color=stress_colors[2])
    ax0.axvspan(stress_changes0[2], stress_changes0[3], color=stress_colors[3])
    ax0.axvspan(stress_changes0[3], stress_changes0[4], color=stress_colors[4])
    ax0.axvspan(stress_changes0[4], Sstart_times[47]+dt.timedelta(seconds=120), color=stress_colors[5])

    ax1.axvspan(Sstart_times[48]-dt.timedelta(seconds=120), stress_changes1[0], color=stress_colors[5])
    ax1.axvspan(stress_changes1[0], stress_changes1[1], color=stress_colors[4])
    ax1.axvspan(stress_changes1[1], stress_changes1[2], color=stress_colors[3])
    ax1.axvspan(stress_changes1[2], stress_changes1[3], color=stress_colors[2])
    ax1.axvspan(stress_changes1[3], stress_changes1[4], color=stress_colors[1])
    ax1.axvspan(stress_changes1[4], Sstart_times[-1]+dt.timedelta(seconds=120), color=stress_colors[0])

    ax0.set_xlim(Sstart_times[0]-dt.timedelta(seconds=120), Sstart_times[47]+dt.timedelta(seconds=120))    
    ax1.set_xlim(Sstart_times[48]-dt.timedelta(seconds=120), Sstart_times[-5]+dt.timedelta(seconds=120))    
    

    ax0.set_ylim(-30, 10)    
    ax1.set_ylim(-30, 10)    


    ticks0 = [ dt.datetime(2021, 10, 7, 10, 0, 0), 
                dt.datetime(2021, 10, 7, 11, 0, 0),
                dt.datetime(2021, 10, 7, 12, 0, 0),
                dt.datetime(2021, 10, 7, 13, 0, 0)]
    ticks1 = [dt.datetime(2021, 10, 8, 7, 30, 0),
                dt.datetime(2021, 10, 8, 8, 30, 0),
                dt.datetime(2021, 10, 8, 9, 30, 0)]
    ax0.set_xticks(ticks0)
    ax1.set_xticks(ticks1)

    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax0.set_xlabel('2021-10-07')
    ax1.set_xlabel('2021-10-08')

    ax0.tick_params(axis='x', which='both', direction='in')
    ax1.tick_params(axis='x', which='both', direction='in')
    ax0.tick_params(axis='y', which='both', direction='in', left=True, labelleft=True)
    ax1.tick_params(axis='y', which='both', direction='in', right=True, labelright=True, left=False, labelleft=False)
    ax1.yaxis.set_label_position("right")


    ax0.set_ylabel(r'dv/v [%]')
    ax1.set_ylabel(r'dv/v [%]')

    legend_elements = [
                    Line2D([0], [0], linestyle='none', label=r'load [kg]:'),
                    Line2D([0], [0], color=rgba1, marker='o', lw=4, linestyle='none', label=r'0'),
                    Line2D([0], [0], color=rgba2, marker='o', lw=4, linestyle='none', label=r'300'),
                    Line2D([0], [0], linestyle='none', label=''),
                    Line2D([0], [0], color=rgba3, marker='o', lw=4, linestyle='none', label=r'600'),
                    Line2D([0], [0], color=rgba4, marker='o', lw=4, linestyle='none', label=r'900'),
                    ]
    ax1.legend(handles=legend_elements, ncol=2, loc='lower left', bbox_to_anchor=(-0.2, 0.1), frameon=True)

    ax0.text(0.9, 1.06,  r'pre-stress [kN]', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.18, 1.01,  r'400', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.41, 1.01,  r'350', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.63, 1.01,  r'300', ha='right', va='bottom', transform=ax0.transAxes)
    ax0.text(0.85, 1.01,  r'250', ha='right', va='bottom', transform=ax0.transAxes)
    ax1.text(0.05, 1.01,  r'250', ha='right', va='bottom', transform=ax1.transAxes)
    ax1.text(0.21, 1.01,  r'300', ha='right', va='bottom', transform=ax1.transAxes)
    ax1.text(0.38, 1.01,  r'350', ha='right', va='bottom', transform=ax1.transAxes)
    ax1.text(0.71, 1.01,  r'400', ha='right', va='bottom', transform=ax1.transAxes)
 
    plt.subplots_adjust(
    top=0.9,
    bottom=0.08,
    left=0.06,
    right=0.94,
    hspace=0.2,
    wspace=0.2
    )
    plt.show()


