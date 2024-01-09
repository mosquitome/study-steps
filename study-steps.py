#==============================================================================
#                           S T U D Y - S T E P S
#==============================================================================

# A pipeline for the bulk import and analsysis of steps data from mosquito 
# electrophys experiments. Still very much in progress. Improvements on other
# approaches include the clustering of peaks in nerve data to assure the same
# peaks are always compared.

# Author        :   David A. Ellis <https://github.com/mosquitome/>
# Organisation  :   University College London

# Requirements  :   [coming soon...]

#                          I N S T R U C T I O N S :
#
# 1. Fill out metadata.txt (tab-delimited).
# 2. Go to 'load multiple files' section below.

#==============================================================================

from sonpy import lib as sp
import matplotlib.pyplot as mplp
import matplotlib.gridspec as mplg
import seaborn as sb
from math import floor
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde, linregress
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import datetime as dt

#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#import hdbscan as hds

pd.options.mode.chained_assignment = None # <- suppress pandas warning

# WARNING: pandas chained assignment warnings have been suppressed (above)...
# WARNING: A file cannot be open in Spike2 during import!
# WARNING: If nerve data is not normalised on import, peak calling will be weird as it calculcate its threshold based on distance from 0 (which is the baseline for normalised nerve data; non-normalised nerve data baselines will vary)
# NOTE: nerve data is recorded at a lower sampling rate than stim and laser data (i.e. arrays are shorter)
# WARNING: Ensure any non-default settings used in call_peaks() are kept the same in cluster_peaks_within() and annotate_stim_onset() (if not, the smoothed nerve data will not be comparible)
# NOTE: Changing the bw_method of the kde for clustering and filtering could effect results.
# NOTE: When importing individual files, forward slash at end of path is important!

def import_steps(directory,max_step='t',max_time=0.1,normalise_nerve=True):
    '''
   
    Import steps data from a directory.
   
    directory       =   a string with the path to a folder containing post-processed (i.e. spike2 scripts have been run) step data.
    max_step        =   a string with the letter of the last step up to which you would like to analyse (default='t').
    normalise_nerve =   boolean of whether you would like to normalise nerve data (i.e. subtract the median; default=True).
   
    '''
    steps = [i for i in 'abcdefghijklmnopqrst'][:'abcdefghijklmnopqrst'.index(max_step)+1]
    data = pd.DataFrame(columns=['directory','type','step','time','value'])
    for i in ['stim','laser','nerve']:
        for jdx,j in enumerate(steps):
            filename = directory + j + '_' + i + '.smr'
            MyFile = sp.SonFile(filename, True)
            #dMaxTime = MyFile.ChannelMaxTime(0)*MyFile.GetTimeBase()
            dSeconds = float(MyFile.ChannelMaxTime(0)*MyFile.GetTimeBase())
            dPeriod = MyFile.ChannelDivide(0)*MyFile.GetTimeBase()
            nPoints = floor(dSeconds/dPeriod)
            time = np.arange(0, nPoints*dPeriod, dPeriod)
            values =  MyFile.ReadFloats(0, nPoints, 0)
            if i=='nerve' and normalise_nerve==True:
                values = values - np.median(values)
            temp = pd.DataFrame({'time':time,'value':values})
            temp['directory'], temp['type'], temp['step'] = directory, i, j
            data = pd.concat([data,temp],axis=0).reset_index(drop=True)
    data = data.loc[data['time']<=max_time]
   
    return data

def call_peaks(time,nerve,frac='infer',threshold=1.5):
    '''
   
    Call peaks in steps nerve data. Returns: 1. original time data, 2. smoothed nerve data, 3. time at peaks (as a list of times), 4. value at peaks (as a list of values), 5. width of peaks corresponding to lists 3 and 4.
   
    time        =   time data as an array of floating point numbers.
    nerve       =   nerve data as an array of floating point numbers.
    frac        =   constant for smoothing function. Must be a float between 0 and 1. 'infer' (default) will work this out to a number I empirically think is good (equivolent of 0.02 for arrays of length 2000; 0.001 doesn't smooth at all | 0.01 removes noise but maintains double peaks present in 'a' steps | 0.1 is very smooth but turns double peaks to one single peak in 'a' steps') based on the sampling rate of input data.
    threshold   =   number of standard deviations away from baseline (median) to call a peak
   
    '''
   
    #print('\nWARNING: Do not use call_peaks() if nerve data was not normalised on import!\n')
   
    if frac=='infer':
        frac = len(time) / 100000.0
    smooth = lowess(nerve,time,is_sorted=True,frac=frac,it=0)
    time2 = smooth[:,0]
    nerve2 = smooth[:,1]
    peaks, _ = find_peaks(nerve2*-1,height=threshold*nerve2.std(),width=0)
    peaktime, peaknerve, peakwidth = time2[peaks], nerve2[peaks], _['widths']
   
    return time2, nerve2, peaktime, peaknerve, peakwidth

def categorise_stim(data):
    '''
   
    Categorise stim data into one of 3 states: -1, 0, or 1 (i.e. fully left, centre, fully right). Returns a copy of the dataframe with a new column called 'stim-categ' describing the current stim category at each time point
   
    data    =   dataframe returned by import_steps().

    '''

    steps = data['step'].unique()
    data2 = data.copy()
    data2['stim-categ'] = np.nan
    for i in steps:
        temp =  data.loc[(data['step']==i) & (data['type']=='stim')]
        state_map1 = {temp['value'].min():-1, np.median(temp['value']):0, temp['value'].max():1}
        state_map2 = {j:min(state_map1.keys(), key=lambda x:abs(j-x)) for j in temp['value']}
        time_map = {j:state_map1[state_map2[temp.loc[temp['time']==j]['value'].values[0]]] for j in temp['time']}
        data2.loc[data2['step']==i,'stim-categ'] = data2.loc[data2['step']==i,'time'].apply(lambda x: time_map[x])
        print('categorising stim ' + i)
    data2['stim-categ'] = data2['stim-categ'].astype(int)
   
    return data2

def cluster_peaks_within(data,bw=0.015):
    '''
   
    Using call_peaks(), identify peaks for each step in dataframe returned by import_steps(), then assign them clusters that are consistent accross all steps. Returns a dataframe with each peak, its related informaion, and which cluster it belongs to.
   
    data    =   dataframe returned by import steps.
    bw      =   bandwidth parameter for kde clustering. Rule of thumb: 0.015=defined, 0.005=broad

    '''
   
    steps = data['step'].unique()
    peaks = pd.DataFrame()
    for i in steps:
        temp = data.loc[(data['type']=='nerve') & (data['step']==i)]
        time, nerve, ptime, pnerve, pwidth = call_peaks(temp['time'],temp['value'])
        temp2 =pd.DataFrame({'peak-time':ptime, 'peak-value':pnerve, 'peak-width':pwidth, 'step':i})
        temp2['peak-id'] = temp.loc[temp['time'].isin(temp2['peak-time'])]['stim-categ'].values # <- initiate peak IDs (note: this only adds broad IDs describing the stim-categ they are associated with)
        for j in ['stim','laser']:
            temp3 = data.loc[(data['type']==j) & (data['step']==i)]
            temp2[j] = temp3.loc[temp3['time'].isin(temp2['peak-time'])]['value'].values
        peaks = pd.concat([peaks,temp2],axis=0).reset_index(drop=True)
       
    kde = gaussian_kde(peaks['peak-time'],bw_method=bw) # defined:0.015, broad:0.05
    s = np.linspace(data['time'].min(),data['time'].max(),1000) # <- make 1000 sample points through the time course
    e = kde(s) # <- find the kernel density at these sample points
    mi, ma = argrelextrema(e,np.less)[0], argrelextrema(e,np.greater)[0] # <- find minima (and maxima) of kernel density estimation (i.e. the divides beteen clusters)
    edges = np.insert(s[mi], [0,len(s[mi])], [data['time'].min(),data['time'].max()]) # <- bin edges to define clusters
    peaks['peak-id'] = peaks['peak-id'].astype(str) + ',' + pd.cut(peaks['peak-time'],labels=range(len(edges)-1),bins=edges).astype(str)
   
    conflicts = peaks.loc[peaks.duplicated(subset=['step','peak-id'])]['peak-id'].unique() # <- identify any peak-ids that have been assigned to more than one peak
    if conflicts.size>0:
        for i in conflicts:
            temp = peaks.loc[peaks['peak-id']==i].copy()
            denominator = max([temp['step'].str.count(j).sum() for j in temp['step'].unique()]) # <- what is the highest number of times the current peak-id has been assigned within a step
            edges2 = np.linspace(temp['peak-time'].min(),temp['peak-time'].max(),denominator+1)
            peaks.loc[temp.index,'peak-id'] = peaks['peak-id'] + pd.cut(temp['peak-time'],labels=range(len(edges2)-1),bins=edges2,include_lowest=True).astype(str) # <- split the peak-id into n new peak-ids, where n=denominator, and assign peaks to these new sub peak-ids
   
    return peaks

def annotate_stim_onset(data):
    '''
   
    Identify when stim changes category (i.e. when was stim onset). Uses output from categorise_stim(). Returns a dataframe with each stim onset time ('time'), stim onset type ('stim-os-type' from where to where e.g. [0,-1]), the corresponding step, stim value ('value') and 'stim-categ'
   
    data    =   dataframe returned by categorise_stim().
   
    '''
    steps = data['step'].unique()
    stims = pd.DataFrame()
    for i in steps:
        temp = data.loc[(data['type']=='stim') & (data['step']==i)].copy()
        temp.loc[temp.index,'stim-onset'] = temp['stim-categ'].diff().apply(lambda x: abs(x)).replace({0:False,np.nan:False,1:True})
        temp2 = temp.loc[temp['stim-onset']==True].copy()
        temp2.loc[temp2.index,'stim-onset-id'] = list(temp.loc[temp2.index-1,'stim-categ'].astype(str).reset_index(drop=True) + ',' + temp2['stim-categ'].astype(str).reset_index(drop=True)) # <- describes what the stim has transitioned from and to ('from,to')
   
        temp3 = data.loc[(data['type']=='nerve') & (data['step']==i)].copy()
        time, nerve, ptime, pnerve, pwidth = call_peaks(temp3['time'],temp3['value'])
        nerve_smooth = interp1d(time,nerve)(temp2['time'].tolist()) # <- smoothed nerve values at times of stim onset (to correspond to smoothed nerve data in peaks)
       
        temp4 = data.loc[(data['type']=='laser') & (data['step']==i)].copy()
        laser = temp4.loc[temp4['time'].isin(temp2['time'])]['value'].values # <- laser values at times of stim onset
       
        temp5 = temp2.loc[:,['directory','step','time','value','stim-categ','stim-onset-id']].rename(columns={'value':'stim-value'})
        temp5 = temp5.assign(**{'nerve-smooth':nerve_smooth,'laser':laser})
        stims = pd.concat([stims,temp5]).reset_index(drop=True)
   
    return stims

def get_zt(metadata,timecolumn):
    '''
   
    Get zt time when steps were performed, and define what broad zt bin it falls in (bins of 30min surrounding zt: 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0). Returns two new arrays (that can be used as columns of metadata)
   
    metadata    =   path to a tab delimited text file desctibing each mosquito and its corresponding analysis. Note: this file has a specific layout and column naming system.
    timecolumn  =   a column in this dataframe with times (as strings in the format HH:MM:SS)
   
    '''
   
    start_time = metadata['experiment-time'].apply(lambda x: dt.datetime.strptime(x,'%H:%M:%S'))
    step_time = metadata[timecolumn].apply(lambda x: dt.datetime.strptime(x,'%H:%M:%S'))
    delta_time = step_time - start_time
    delta_zt = delta_time.apply(lambda x: x.total_seconds() / 3600.0) # <- get total seconds as a proportion of an hour
    step_zt = metadata['experiment-zt'] + delta_zt
    bin_zt = pd.cut(step_zt, np.arange(-0.25,24.25,0.5), labels=np.arange(0,24,0.5), right=False)
   
    return step_zt, bin_zt

def pad_na(target,pad_length,pad_material=np.nan):
    '''

    pads lists evenly (or nearly evenly for an odd pad_length) on either side with specified unit (default NA).
   
    target          =   list to pad
    pad_length      =   how much pad to add
    pad_material    =   unit to pad with (default NA)

    '''
    target2 = target.copy()
    for idx,i in enumerate([pad_material]*pad_length):
        if idx%2==0:
            target2.insert(0,i)
        else:
            target2.insert(len(target2),i)
   
    return target2

def fit_sin(time,y):
    '''
   
    fit sine curve to the input time sequence, and return a dictionary of fitting parameters 'amp', 'omega', 'phase', 'offset', 'freq', 'period' and 'fitfunc'. to use the fitfunc, a timeseries needs to be supplied: dictionary['fitfunc'](timeseries).
   
    time    =   time series to fit to (list-like; must be uniformly spaced)
    y       =   y data to fit to (list-like)
   
    '''
    tt = np.array(time)
    yy = np.array(y)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
   
    return {'amp': A, 'omega': w, 'phase': p, 'offset': c, 'freq': f, 'period': 1./f, 'fitfunc': fitfunc, 'maxcov': np.max(pcov), 'rawres': (guess,popt,pcov)}

def get_xstats(data,stim_categ='both',frac='infer',suppress_plots=True,window=20,xsteady_method='polynomial'):
    '''

    Gets xpeak and xsteady values. Units: Displacement e.g. xsteady (nM), velocity (), acceleration (nM/s**2), force (pN)
   
    data            =   dataframe returned by categorise_stim().
    stim_categ      =   which step direction to use. (i.e. stim category). Must be either 'both', or one of the following integers: 1, -1.
    frac            =   constant for smoothing function. Must be a float between 0 and 1. 'infer' (default) will work this out to a number I empirically think is good (equivolent of 0.025 for arrays of length 2500; 0.001 doesn't smooth at all | 0.01 removes noise but maintains double peaks present in 'a' steps | 0.1 is very smooth but turns double peaks to one single peak in 'a' steps') based on the sampling rate of input data.
    suppress_plots  =   boolean of whether to plot each step with overlayed xpeak, tpeak, xsteady and tsteady
    window          =   number of data points per-window for measurement of slope from time-velocity data (to obtain acceleration). Default=20
    xsteady_method  =   method used to estimate xsteady. Default ('polynomial') is to fit a polynomial through the data during forcing, then take the point where that polynomial plateaus (i.e. rate of change is minimum). Alternatives are 'end-fit' (take the average of a single sinusoid wave at the end of the force step), ''
   
    '''
    # Note: for everything with -1 stim, absolute vallues are taken so that troughs are converted to peaks for peak calling. They are converted back afterwards
    mass = 5.414 * 1e-6 # 1e-20 # <- apparent mass (in grams) of an anopheles flaggelum (see Su et al. 2018?)
    steps = data['step'].unique()
    laser = pd.DataFrame()
    if stim_categ=='both':
        sc = [-1,1]
    else:
        sc = [stim_categ]
    for s in sc:
        for i in steps:
            print('retrieving xstats for current file, stimulus',s,', step',i)
            temp = data.loc[(data['type']=='laser') & (data['step']==i) & (data['stim-categ']==s)].copy()
            if len(temp)==0:
                continue # <- skip to next step if no data is available for current one
            if s==-1:
                temp['value'] = temp['value'].apply(lambda x: abs(x) if x<0 else (-abs(x) if x>0 else 0)) # <-if stimulus is negative, convert data to absolute (so that peak calling works)
            if frac=='infer':
                frac2 = 62.5 / len(temp) # len(temp) / 100000.0
            else:
                frac2 = frac
            if not 0 <= frac2 <= 1:
                continue # <- skip to next step if frac is strange
            smooth = lowess(temp['value'],temp['time'],is_sorted=True,frac=frac2,it=0)
            time2 = smooth[:,0]
            laser2 = smooth[:,1]
       
            #velo2 = [linregress(time2[j:j+2],laser2[j:j+2])[0] for j in range(len(laser2)-1)] # <- calculate velocity continuously along data by getting the slope of time-vs-laser (smoothed but not best fit (below)) of every neighbouring point
            #acce2 = [linregress(time2[j:j+2],velo2[j:j+2])[0] for j in range(len(velo2)-1)] # <- calculate acceleration continuously along data by getting the slope of velocity-vs-laser (smoothed but not best fit (below)) of every neighbouring point
       
            peaks_laser, _ = find_peaks(laser2,width=0) # <- identifies peaks throughout laser data, not just the initial peak (i.e. not just xpeak)
            if len(peaks_laser)==0:
                continue # <- # <- skip to next step if no data is available for current one
            xpeak = temp['value'].tolist()[peaks_laser[0]]
            tpeak = temp['time'].tolist()[peaks_laser[0]] # <- time of xpeak, relative to stim-onset (stim-onset being the same as the begining of the temp dataframe)
           
            if s==1:
                low = 0
            elif s==-1:
                low = temp['time'].min() - 0.004
            high = temp['time'].max() + 0.004
            temp2 = data.loc[(data['type']=='laser') & (data['step']==i) & (data['time']>low) & (data['time']<high)] # <- import another temp in including a period before stim-onset
            if s==-1:
                temp2['value'] = temp2['value'].apply(lambda x: abs(x) if x<0 else (-abs(x) if x>0 else 0)) # <- if stimulus is negative, convert to absolute so that peak calling works
           
            smooth = lowess(temp2['value'],temp2['time'],is_sorted=True,frac=frac2,it=0)
            time3 = smooth[:,0]
            laser3 = smooth[:,1]
       
            velo3 = [linregress(time3[j:j+window],laser3[j:j+window])[0] for j in range(len(laser3)-window-1)] # <- calculate velocity continuously along data by getting the slope of time-vs-laser (smoothed but not best fit (below)) of neighbouring points in windows of size window (default=20)
            acce3 = [linregress(time3[j:j+window],velo3[j:j+window])[0] for j in range(len(velo3)-window-1)] # <- calculate acceleration continuously along data by getting the slope of velocity-vs-laser (smoothed but not best fit (below)) of neighbouring points in windows of size window (default=20)
            velo3 = pad_na(velo3,len(time3)-len(velo3)) # <- pad velo3 on either side with NAs
            acce3 = pad_na(acce3,len(time3)-len(acce3)) # <- pad acce3 on either side with NAs
           
            tdx = np.where(time3<=temp['time'].min())[0].max() # <- index position of time of stim-onset
            tdx2 = np.where(time3>=temp['time'].max())[0].min() # <- index position of time of stim-end
            time3b = time3[tdx:tdx2]
            acce3b = acce3[tdx:tdx2]
            peaks_acce, _ = find_peaks(acce3b,width=0) # <- identifies peaks throughout acceleration data after stim-onset, not just the initial peak (i.e. not just acce_onset)
           
            acce_onset = acce3b[peaks_acce[0]] # <- the recievers initial acceleration (i.e. the peak in acceleration just before xpeak)
            tacce_onset = time3b[peaks_acce[0]] # <- time of acce_onset
            if tacce_onset==time3[0]:
                acce_onset = acce3b[peaks_acce[1]] # <- if acce onset is artefactually at the very start, use the second peak called in the data instead of the first
                tacce_onset = time3b[peaks_acce[1]]
            if tacce_onset>tpeak:
                acce_onset = np.nan # <- if the only peak identified to use as acce_onset is very late in the data set, acce_onset was not possible to call
                tacce_onset = np.nan
            force = acce_onset * mass # <- external force; calculated using the constant mass (and scaling factor to adjust magnitude of unit) specified at start of function
           
            if xsteady_method=='polynomial':
                f = np.poly1d(np.polyfit(temp['time'].tolist(),temp['value'].tolist(),3))
                laser4 = f(time2).astype(float) # <- three-degree polynomial line of best fit through data
                velo4 = [linregress(time2[j:j+2],laser4[j:j+2])[0] for j in range(len(laser4)-1)] # <- calculate velocity continuously along data by getting the slope of time-vs-laser (smoothed) of every neighbouring point
                #acce4 = [linregress(time2[j:j+2],velo4[j:j+2])[0] for j in range(len(velo4)-1)] # <- calculate acceleration continuously along data by getting the slope of velocity-vs-laser (smoothed) of every neighbouring point
           
                zeros = np.argwhere(np.diff(np.sign(np.zeros(len(velo4))-velo4))).flatten() # <- all points where the velocity crosses zero in the time-velocity curve
                if len(zeros)==0:
                    if np.argmin(np.abs(velo4))==len(velo4)-1: # <- 1a) when the velocity never reaches zero, and the slowest velocity is the final value (i.e. velocity has not plateau'd)
                        xsteady = np.nan
                        tsteady = np.nan
                    elif np.argmin(np.abs(velo4))==0: # <- 1b) when the velocity never reaches zero, and the slowest velocity is the first value (i.e. velocity has not plateau'd)
                        xsteady = np.nan
                        tsteady = np.nan
                    else: # <- 2) when the velocity never reaches zero, but does plateau
                        xsteady = laser4[np.argmin(np.abs(velo4))]
                        tsteady = temp['time'].tolist()[np.argmin(np.abs(velo4))] # <- time xsteady is reached, relative to stim-onset
                elif len(zeros)==1: # <- 3) when the velocity reaches zero at a single point in time
                    xsteady = laser4[np.argmin(np.abs(velo4))]
                    tsteady = temp['time'].tolist()[np.argmin(np.abs(velo4))]
                else: # <- 4) when the velocity reaches zero multiple times (due to initial artefacts in the polynomial)
                    xsteady = laser4[zeros[-1]] # <- use the last zero crossing (as the initial ones tend to be artefacts)
                    tsteady = temp['time'].tolist()[zeros[-1]]
           
            elif xsteady_method=='end-fit':
                f = fit_sin(temp['time'],temp['value'])
                p = f['period'] # <- period of fitted sine wave, i.e. the length of one cycle, in time units
                t = temp.loc[temp['time']>=temp['time'].max()-p]['time'] # <- extract time values for one cycle at the end of the step
                f['fitfunc'](t.astype(float))
           
            if s==-1:
                xpeak, xsteady, acce_onset, force = -abs(xpeak), -abs(xsteady), -abs(acce_onset), -abs(force)
           
            if suppress_plots==False:
               
                if s==-1:
                    temp2 = data.loc[(data['type']=='laser') & (data['step']==i) & (data['time']>low) & (data['time']<high)] # <- import another temp in including a period before stim-onset
                    laser2 = [abs(x) if x<0 else -abs(x) if x>0 else 0 for x in laser2] #-abs(laser2)
                    laser4 = [abs(x) if x<0 else -abs(x) if x>0 else 0 for x in laser4]
                    acce3 = [abs(x) if x<0 else -abs(x) if x>0 else 0 for x in acce3]
                    acce3b = [abs(x) if x<0 else -abs(x) if x>0 else 0 for x in acce3b]
                   
                temp3 = data.loc[(data['type']=='stim') & (data['step']==i) & (data['time']>low) & (data['time']<high)] # <- import another stim data to plot
               
                fig = mplp.figure()
                ax = fig.add_subplot(111)
                ax.plot(temp2['time'],temp2['value'],c='lightgrey')
                ax.plot(time2,laser2,c='black')
                ax.plot(temp['time'],laser4,c='red')
                ax.axhline(xpeak,c='black',ls=':')
                ax.axvline(tpeak,c='black',ls=':')
                ax.axhline(xsteady,c='red',ls=':')
                ax.axvline(tsteady,c='red',ls=':')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Displacement (nM)')
                ax.set_xlim(xmin=low,xmax=high)
               
                fig = mplp.figure()
                ax = fig.add_subplot(111)
                ax.plot(time3,acce3,c='lightgrey')
                ax.plot(time3b,acce3b,c='black')
                ax.axhline(acce_onset,c='blue',ls=':')
                ax.axvline(tacce_onset,c='blue',ls=':')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(r'Acceleration (m/$s^2$)')
                ax.set_xlim(xmin=low,xmax=high)
               
                fig = mplp.figure()
                ax = fig.add_subplot(111)
                ax.plot(temp3['time'],temp3['value'],c='black')
                ax.axvline(tpeak,c='black',ls=':',alpha=0.5)
                ax.axvline(tsteady,c='red',ls=':',alpha=0.5)
                ax.axvline(tacce_onset,c='blue',ls=':',alpha=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Stimulus (V)')
                ax.set_xlim(xmin=low,xmax=high)
           
            temp3 = pd.DataFrame([[i,xpeak,tpeak,xsteady,tsteady,acce_onset,tacce_onset,force]],columns=['step','xpeak','tpeak','xsteady','tsteady','acce-onset','tacce-onset','force'])
            laser = pd.concat([laser,temp3]).reset_index(drop=True)
       
    return laser

def import_master(metadata,drop_flagged=True,sub_directory='/steps/'):
    '''
   
    Given a metadata file for an experiment, import steps data, identify stim onsets, call nerve peaks and cluster them within a sample. Returns three dictionaries with informaion on stim, laser and nerve. These dictionaries contain a dataframe for each mosquito with summary statistics for each attribute (returned by annotate_stim_onset(), _(), and cluster_peaks_within(), respectively; note: addtional columns included: mosquito-genotype, mosquito-sex, mosquito-drug, mosquito-age, mosquito-id, steps-zt, steps-temperature, steps-state)
   
    metadata        =   path to a tab delimited text file desctibing each mosquito and its corresponding analysis. Note: this file has a specific layout and column naming system.
    drop_flagged    =   whether mosquitoes marked in the 'flag-steps' column should be excluded (default is True).
    sub_directory   =   string describing the universal path within each individual's directory (described in metadata) to a subdirectory containing the post-processed steps data (default for steps analysis is '/steps/')

    '''
   
    metadata = pd.read_table(metadata,skiprows=[0,1])
    if drop_flagged==True:
        metadata = metadata.loc[metadata['flag-steps']==False]
    metadata['steps-zt'], metadata['steps-zt-bin'] = get_zt(metadata,'steps-time')
   
    stim = {}
    laser = {}
    nerve = {}
    for idx,i in metadata.iterrows():
        print('importing ' + i['mosquito-id'])
        data = import_steps(i['file-path']+sub_directory)
        data = categorise_stim(data)
        stim[i['mosquito-id']] = annotate_stim_onset(data)
        laser[i['mosquito-id']] = get_xstats(data)
        nerve[i['mosquito-id']] = cluster_peaks_within(data)
        for j in ['mosquito-genotype','mosquito-sex','mosquito-drug','mosquito-age','mosquito-id','steps-zt','steps-zt-bin','steps-temperature','steps-state']:
            stim[i['mosquito-id']][j] = i[j]
            laser[i['mosquito-id']][j] = i[j]
            nerve[i['mosquito-id']][j] = i[j]

    return stim, laser, nerve

def filter_peaks(peaks_master,tmin=0.0,tmax=0.1,sfilter=1,nfilter=1,suppress_plots=False,bw=0.015):
    '''
   
    Filter peaks from dataframes produced by cluster_peaks_between(). Peaks identified too few times within an individual, or at too few force steps (within an individual) can be filtered. A filtered peaks dataframe is returned with a new column - 'peak-id-global'. These are not the final clustered peak-ids, but are used to cluster peaks for filtration.

    peaks_master        dataframe produced by cluster_peaks_between(). This filtering function is applied within cluster_peaks_between().
    tmin, tmax      =   start and end time of data
    sfilter         =   integer value of the lowest number of steps a peak must be called in within an individual for it to pass filtering for that individual (can be set to 0 to remove this filtering step)
    nfilter         =   integer value of the lowest number of individuals a peak must be found in for it to pass filtering (can be set to 0 to remove this filtering step)
    bw              =   bandwidth parameter for kde clustering. Rule of thumb: 0.015=defined, 0.005=broad

    '''
    peaks2 = peaks_master.copy()
    filtered = False
    count = 0
    while filtered==False:
        print('\n \t ... filtering iteration',count,'...')
        count += 1
        kde = gaussian_kde(peaks2['peak-time-avg'],bw_method=bw) # defined:0.015, broad:0.05
        s = np.linspace(tmin,tmax,1000) # <- make 1000 sample points through the time course
        e = kde(s) # <- find the kernel density at these sample points
        mi, ma = argrelextrema(e,np.less)[0], argrelextrema(e,np.greater)[0] # <- find minima (and maxima) of kernel density estimation (i.e. the divides beteen clusters)
        edges = np.insert(s[mi], [0,len(s[mi])], [tmin,tmax]) # <- bin edges to define clusters
        peaks2['peak-id-global'] = pd.cut(peaks2['peak-time-avg'],labels=range(len(edges)-1),bins=edges).astype(str) # <- give a working id to each peak
       
        if suppress_plots==False:
            fig = mplp.figure()
            ax = fig.add_subplot(111)
            ax.plot(s,e)
            ax.scatter(s[ma],e[ma])
            ax.annotate('cluster identities, filtering iteration '+str(count), [0,ax.get_ylim()[1]+ax.get_ylim()[1]/15.0])
            ax.set_ylim(ymax=ax.get_ylim()[1]+ax.get_ylim()[1]/5.0)
       
        if sfilter==0:
            temp = peaks2.copy()
        else:
            sfilter2 = 4 + (7 * sfilter)
            temp = peaks2[peaks2.fillna(0).astype(bool).sum(axis=1)>sfilter2] # <- subset of peaks found more than sfilter times in any given individual (the number 4 + 7 refers to the four non-step columns (e.g. mosquito-id) plus one set of seven step columns for a given step i.e. any row with only eleven non-zero cells must only have data for a single step)
            sfilter = 0
        nunq = [i for i in temp['peak-id-global'].unique() if len(temp.loc[temp['peak-id-global']==i]['mosquito-id'].unique()) > nfilter] # <- get a list of peaks from the subset above found in at least nfilter individuals (in order to drop peaks found in only one sample)
        if len(nunq)==len(peaks2['peak-id-global'].unique()):
            filtered = True
        else:
            peaks2 = peaks2.loc[peaks2['peak-id-global'].isin(nunq)]
            print('\n... reclustering for filtration ...')
   
    return peaks2


def cluster_peaks_between(nerve_master,stim_master,subset={'steps-state':'so'},impute=False,tmin=0.0,tmax=0.1,sfilter=1,nfilter=1,bw=0.2):
    '''
   
    Using various information about peaks collated from all samples, cluster peaks across all samples using multidimensional clustering. Requires nerve dictionary returned by import_master(). Returns a dataframe with one row for each within-sample peak, and a column for each descriptor of that peak at each step. Note that peaks before any stimulus are automatically filtered.
   
    nerve_master    =   dictionary returned by import_master() with one cluster_peaks_within() dataframe for each sample. Dictionary keys are individual sample names.
    stim_master     =   dictionary returned by import_master() with one annotate_stim_onset() dataframe for each sample. Dictionary keys are individual sample names.
    subset          =   to perform analyses on only a subset of samples, state a dictionary of column headers in each sample's cluster_peaks_within() dataframe and corresponding values by which to subset. If you do not want to subset, set subset=None
    impute          =   whether to impute values for variables at a step where that peak was not found (peak value and width are 0 as the peak was not found, peak time is based on the median time of all peaks with that peak-id (stim and laser are not imputed - if this were to be implemented, it would be good to use the corresponding values at that peak-time for the current step)
    tmin, tmax      =   start and end time of data
    sfilter         =   integer value of the lowest number of steps a peak must be called in within an individual for it to pass filtering for that individual (can be set to 0 to remove this filtering step)
    nfilter         =   integer value of the lowest number of individuals a peak must be found in for it to pass filtering (can be set to 0 to remove this filtering step)
    bw              =   bandwidth parameter for kde clustering. Rule of thumb (different from previous clustering): broad=0.2
   
    '''
    values = ['peak-time','peak-value','peak-width','stim','laser']
    peaks2 = pd.DataFrame()
    for i in nerve_master.keys():
        print(i)
        temp = nerve_master[i]
        if subset!=None:
            for j in subset.keys():
                temp = temp.loc[temp[j]==subset[j]]
        peaks = temp['peak-id'].unique()
        steps = temp['step'].unique()
        for j in peaks:
            temp2 = temp.loc[temp['peak-id']==j]
           
            if temp2['peak-time'].median() < stim_master[i]['time'].min(): # to filter any peaks that fall before stimuli have even begun:
                continue
           
            peak = pd.DataFrame([[i,j]],columns=['mosquito-id','peak-id-within']) # <- begin making a row of a data for the current peak
            for k in steps:
                if k in temp2['step'].tolist():
                    temp3 = temp2.loc[temp2['step']==k,values].rename(columns={x:k+'-'+x for x in values}).reset_index(drop=True)
                elif impute==True:
                    temp3 = pd.DataFrame([[0]*len(values)],columns=[k+'-'+x for x in values]).reset_index(drop=True)
                    temp3[k+'-peak-time'] = temp.loc[temp['peak-id']==j]['peak-time'].median()
                    temp3[k+'-stim'], temp3[k+'-laser'] = np.nan, np.nan
                else:
                    temp3 = pd.DataFrame([[np.nan]*len(values)],columns=[k+'-'+x for x in values]).reset_index(drop=True)
                peak = pd.concat([peak,temp3],axis=1)
               
                temp4 = stim_master[i].loc[stim_master[i]['step']==k]
                if not temp.loc[(temp['peak-id']==j) & (temp['step']==k)]['peak-time'].values.tolist(): # for any missing values (i.e. cases where imputation might be used above):
                    soi = np.nan
                    dso = np.nan
                else:
                    pti = temp.loc[(temp['peak-id']==j) & (temp['step']==k)]['peak-time'].values[0] # <- time of peak (pti=peak-time)
                    soi = pd.cut([pti],temp4['time'].tolist()+[tmax],include_lowest=True,labels=temp4['stim-onset-id'],ordered=False)[0] # <- find which stim the peak is responding to (soi=stim-onset-id)
                    dso = pti - temp4.loc[temp4['stim-onset-id']==soi,'time'].values[0] # <- calculate the time since this stimulus onset (this will give an estimate of latency, and also to align peaks later on; dso=delta-stim-onset)
                peak[k+'-stim-onset-id'], peak[k+'-delta-stim-onset'] = soi, dso
               
            peaks2 = pd.concat([peaks2,peak]).reset_index(drop=True)
    peaks2['peak-time-avg'] = peaks2.loc[:, [i for i in peaks2.columns if 'peak-time' in i]].mean(axis=1) # <- average time of each peak across all steps
    peaks2 = filter_peaks(peaks2,tmin=tmin,tmax=tmax,sfilter=sfilter,nfilter=nfilter,suppress_plots=True)    
       
    deltas = [peaks2[i+'-delta-stim-onset'].dropna().tolist() for i in [j.split('-')[0] for j in peaks2.columns if j.split('-')[1]=='delta']]
    deltas = [i for j in deltas for i in j]
   
    kde = gaussian_kde(deltas,bw_method=bw) # no longer same rule of thumb as otehr clustering (i.e. defined:0.015, broad:0.05)
    smin, smax = min(deltas) - (max(deltas)-min(deltas))*0.05, max(deltas) - (max(deltas)-min(deltas))*0.05 # <- low and high bounds for kde clustering (from 5% of the delta range below the lowest, to 5% above the highest delta)
    s = np.linspace(smin,smax,1000)
    e = kde(s)
    mi, ma = argrelextrema(e,np.less)[0], argrelextrema(e,np.greater)[0]
    edges = np.insert(s[mi], [0,len(s[mi])], [smin,smax]) # <- bin edges to define clusters
   
    # make a new peak-id stating which stim-onset the peak is in response to, and of those peaks, which cluster it belongs to.
    steps = [i.split('-')[0] for i in peaks2.columns if '-peak-time' in i]
    for idx,i in peaks2.iterrows():
        peaks2.loc[idx,'peak-id-global-2'] = i[[j+'-stim-onset-id' for j in steps]].dropna().values[0] # <- initialise ids with the stim onset (the sub-cluster of the peak will then be added below, once the cluster boundaries have been defined)
        peaks2.loc[idx,'delta-stim-onset-avg'] = i[[j+'-delta-stim-onset' for j in steps]].median() # <- calculate an average (median) delta-stim-onset for each peak accross all steps.
    peaks2['peak-id-global-2'] = '[' + peaks2['peak-id-global-2']
    peaks2['peak-id-global-2'] += '],' + pd.cut(peaks2['delta-stim-onset-avg'],labels=range(len(edges)-1),bins=edges).astype(str) # <- update stim ids with their subcluster (based on how near to the stim-onset it is). This clustering uses ALL delta-stim-onset data i.e. all peaks at all stims, normalised.

    print('\n',str(len(peaks2['peak-id-global-2'].unique())),'peaks identified')

    fig = mplp.figure()
    ax = fig.add_subplot(111)
    ax.hist(peaks2['delta-stim-onset-avg'],bins=50)
    ymax1 = ax.get_ylim()[1]
    ax2 = fig.add_subplot(111,sharex=ax,frame_on=False)
    ax2.yaxis.tick_right()
    ax2.plot(s,e,c='red')
    ax2.scatter(s[ma],e[ma],c='red')
    for i,j in zip(peaks2['peak-id-global-2'].unique(),np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],len(peaks2['peak-id-global-2'].unique()))):
        temp = peaks2.loc[peaks2['peak-id-global-2']==i]
        ax.annotate(i,[temp['delta-stim-onset-avg'].median(),j],ha='center')
    for j in mi:
        ax.axvline(s[j],ls=':',c='grey')
    #ax.set_ylim(ymax=ax.get_ylim()[1]+ax.get_ylim()[1]/5.0)
   
    standard_clusters = ['[0,1],0','[0,1],1','[0,1],2','[0,1],3','[1,0],0','[1,0],1','[1,0],2','[1,0],3','[0,-1],0','[0,-1],1','[0,-1],2','[0,-1],3','[-1,0],0','[-1,0],1','[-1,0],2','[-1,0],3']
    fig = mplp.figure()
    ax = fig.add_subplot(111)
    if set(standard_clusters)==set(peaks2['peak-id-global-2'].unique()):
        sb.stripplot(ax=ax,data=peaks2,x='delta-stim-onset-avg',y='peak-id-global-2',order=standard_clusters)
    elif set(standard_clusters).issubset(set(peaks2['peak-id-global-2'].unique())):
        standard_clusters += set(peaks2['peak-id-global-2'].unique()) - set(standard_clusters)
        sb.stripplot(ax=ax,data=peaks2,x='delta-stim-onset-avg',y='peak-id-global-2',order=standard_clusters)
    else:
        sb.stripplot(ax=ax,data=peaks2,x='delta-stim-onset-avg',y='peak-id-global-2')
   
    return peaks2

#==============================================================================
#                  L O A D   M U L T I P L E   F I L E S
#==============================================================================

#metadata = 'Z:/David/serotonin_project/laser/README_preliminary_recordings.txt'
metadata = 'F:/Picrotoxina_data_Dave/metadata.txt'
stim, laser, nerve = import_master(metadata, sub_directory='/')
#stim, laser, nerve = import_master(metadata)
peaks = cluster_peaks_between(nerve,stim,subset=None)
#peaks = cluster_peaks_between(nerve,stim) # <- only SO data
#peaks['genotype'] = peaks['mosquito-id'].apply(lambda x: x.split('_')[0])

#==============================================================================
#                   V I E W   S I N G L E   F I L E
#==============================================================================

directory = 'F:/Picrotoxina_data_Dave/ZT12/PIC_1mM/Anopheles_gambiae_G3_male_5-7_days_old_virgin_ZT12_mos_1_01_06_21_fibrillae_erected_Quiesc_used_sweeps_done/Cal_baseline_done/'
directory = 'F:/Picrotoxina_data_Dave/ZT12/PIC_1mM/Anopheles_gambiae_G3_male_5-7_days_old_virgin_ZT12_mos_1_27_05_21_fibrillae_erected_SSO_used_sweeps_done/Cal_baseline_done/'
directory = 'F:/Picrotoxina_data_Dave/ZT12/PIC_1mM/Anopheles_gambiae_G3_male_5-7_days_old_virgin_ZT12_mos_1_24_08_21_fibrillae_erected_quiesc_used/Cal_baseline_done/'
#directory = 'Z:/David/serotonin_project/laser/wt_inverted/wt_male_3/steps/'
#directory = 'Z:/David/serotonin_project/laser/wt_inverted/wt_male_8/steps/' # <- many anomalous results in laser data
data = categorise_stim(import_steps(directory))
# laser_sub = get_xstats(data,suppress_plots=False)
# peaks_sub = cluster_peaks_within(data)
# stims_sub = annotate_stim_onset(data)

# NERVE:

ymax = data.loc[data['type']=='nerve']['value'].max()
ymin = data.loc[data['type']=='nerve']['value'].min()
ymax += (ymax - ymin) * 0.05
ymin -= (ymax - ymin) * 0.05
for i in steps:
    temp = data.loc[(data['step']==i) & (data['type']=='nerve')]
    time2, nerve2, tpeaks, npeaks, _ = call_peaks(temp['time'],temp['value'])
    #temp2 = pd.DataFrame({'peak-time':tpeaks, 'peak-value':npeaks, 'step':i})
    #peaks = pd.concat([peaks,temp2],axis=0).reset_index(drop=True)
   
    fig = mplp.figure()
    ax = fig.add_subplot(111)
    ax.plot(temp['time'],temp['value'],c='blue')
    ax.plot(time2,nerve2,c='red')
    ax.axhline(-2.0 * nerve2.std(),ls=':')
    for j in tpeaks:
        ax.axvline(x=j)
    #ax.set_ylim(ymin=ymin,ymax=ymax)

#==============================================================================
#                       N E R V E   A N A L Y S I S
#==============================================================================

# 1. Analyse quiescent and SSOing mosquitoes seperately (but when importing in the future, check the bw parameter for clustering)?
# 2. For analyses of CAP amplitudes (both raw and normalised), it makes more sense to look at data from the 1st peak at each stim-onset ('[i,j],0') as this is the biggest peak and the nerves probably belong to the jo being stimulated. Data from all of these appear comparible - I will choose the nicest example.

colours = {'4223':'#7fc97f','11481':'#beaed4','4222':'#fdc086','wt':'black'}

#order = ['[0,1],0','[0,1],1','[0,1],2','[0,1],3','[1,0],0','[1,0],1','[1,0],2','[1,0],3','[0,-1],0','[0,-1],1','[0,-1],2','[0,-1],3','[-1,0],0','[-1,0],1','[-1,0],2','[-1,0],3']
order = ['[0,1],0','[0,1],1','[0,1],2','[1,0],0','[1,0],1','[1,0],2','[0,-1],0','[0,-1],1','[0,-1],2','[-1,0],0','[-1,0],1','[-1,0],2',]
#sb.stripplot(data=peaks,x='peak-id-global-2',y='delta-stim-onset-avg',hue='genotype',dodge=True,order=order)
#sb.boxplot(data=peaks,x='peak-id-global-2',y='delta-stim-onset-avg',hue='genotype',order=order)
#mplp.legend([],[], frameon=False)

#

p = [] # <- peaks
c = [] # <- colours
o = [] # <- line offsets
w = [] # <- line widths (keep the same for every single raster line)
count = 0
for step in 'abcdefghijklmnopqrst':
    for i in peaks['genotype'].unique():
        temp = peaks.loc[peaks['genotype']==i]
        for j in temp['mosquito-id'].unique():
            temp2 = temp.loc[temp['mosquito-id']==j]
            p.append(temp2[step+'-delta-stim-onset'].dropna())
            c.append(colours[i])
            o.append(count)
            w.append(2)
            count -= 1
fig = mplp.figure(figsize=[6,4])
ax = fig.add_subplot(111)
ax.eventplot(p,lineoffsets=o,colors=c,linewidths=w)
ax.set_xlabel('Time since stimulus onset (ms)')
ax.set_ylabel('Force steps')
ax.set_yticks(np.linspace(0,count,20))
ax.set_yticklabels([i for i in 'abcdefghijklmnopqrst'])

histogram = pd.DataFrame(columns=['genotype','mosquito-id','step','delta-stim-onset'])
for i in peaks['genotype'].unique():
    temp = peaks.loc[peaks['genotype']==i]
    for j in temp['mosquito-id'].unique():
        temp2 = temp.loc[temp['mosquito-id']==j]
        temp3 = pd.DataFrame(columns=['genotype','mosquito-id','step','delta-stim-onset'])
        for step in 'abcdefghijklmnopqrst':
            temp4 = pd.DataFrame(columns=['genotype','mosquito-id','step','delta-stim-onset'])
            temp4['delta-stim-onset'] = temp2[step+'-delta-stim-onset'].dropna()
            temp4['genotype'], temp4['mosquito-id'], temp4['step'] = i, j, step
            temp3 = pd.concat([temp3,temp4]).reset_index(drop=True)
        histogram = pd.concat([histogram,temp3]).reset_index(drop=True)
histogram['delta-stim-onset'] = histogram['delta-stim-onset'].astype(float)

#sb.histplot(data=histogram,x='delta-stim-onset',hue='genotype',palette=colours,\
#            bins=60,hue_order=['11481','4223','wt']) #fill=False

#sb.histplot(data=histogram,x='delta-stim-onset',y='genotype',hue='genotype',palette=colours,\
#            bins=75)

#fig = mplp.figure(figsize=[6,3])
#ax = fig.add_subplot(111)
#sb.stripplot(data=histogram,x='delta-stim-onset',y='genotype',hue='genotype',alpha=0.5,\
#             jitter=0.3,palette=colours,ax=ax)
#ax.set_ylabel('')
#ax.set_xlabel('Time since stimulus onset (ms)')

#sb.kdeplot(data=histogram,x='delta-stim-onset',hue='genotype',palette=colours,bw_adjust=0.3,fill=True)

xmin, xmax = histogram['delta-stim-onset'].min(), histogram['delta-stim-onset'].max()
fig = mplp.figure(figsize=[5,2])
gs = mplg.GridSpec(3,1,hspace=-0.4)
ax = {}
for idx,i in enumerate(['wt','4223','11481']):
    temp = histogram.loc[histogram['genotype']==i]
    ax[i] = fig.add_subplot(gs[idx])
    #sb.histplot(ax=ax[i],data=temp,x='delta-stim-onset',color=colours[i],bins=60)
    sb.kdeplot(ax=ax[i],data=temp,x='delta-stim-onset',color='white',fill=True,\
               bw_adjust=0.3,alpha=1,zorder=idx,lw=5)
    sb.kdeplot(ax=ax[i],data=temp,x='delta-stim-onset',color=colours[i],fill=True,\
               bw_adjust=0.3,zorder=idx)
    ax[i].set_xlim(xmin=xmin,xmax=xmax)
    sb.despine(ax=ax[i],bottom=True,left=True)
    ax[i].tick_params(axis='both',bottom=False,labelbottom=False,left=False,labelleft=False)
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    ax[i].patch.set_alpha(0.0)
    ax[i].annotate(i,[xmax*0.99,ax[i].get_ylim()[1]*0.2],color=colours[i],ha='right')
ymax = ax['11481'].get_ylim()[1]
for i in ['wt','4223','11481']:
    temp = peaks.loc[peaks['genotype']==i].copy()
    temp['_'] = temp['peak-id-global-2'].apply(lambda x: x.split(',')[2])
    for j in ['0','1','2']:
        temp2 = temp.loc[temp['_']==j].copy()
        ax['11481'].vlines(x=temp2['delta-stim-onset-avg'].median(),ymin=0,ymax=900,\
                           color=colours[i],ls='--',clip_on=False)
ax['11481'].set_ylim(ymax=ymax)
sb.despine(ax=ax['11481'],bottom=False,left=True)
ax['11481'].tick_params(axis='both',bottom=True,labelbottom=True,left=False,labelleft=False)
ax['11481'].set_xlabel('Time since stimulus onset (ms)')
ax['4223'].set_ylabel('Density')

#

#temp = peaks.loc[peaks['peak-id-global-2']=='[0,1],0']
#temp = peaks.loc[peaks['peak-id-global-2']=='[1,0],0']
#temp = peaks.loc[peaks['peak-id-global-2']=='[0,-1],0']
temp = peaks.loc[peaks['peak-id-global-2']=='[-1,0],0']
#temp = peaks.loc[peaks['peak-id-global-2']=='[0,1],2']
#temp = peaks.loc[peaks['peak-id-global-2']=='[0,-1],2']
temp['max-cap'] = temp[[i+'-peak-value' for i in 'abcdefghijklmnopqrst']].min(axis=1)

summary = pd.DataFrame()
for i in ['wt','11481','4223']:
    temp2 = temp.loc[temp['genotype']==i]
    for j in 'abcdefghijklmnopqrst':
        temp3 = pd.DataFrame()
        temp3['prop-max-cap'] = temp2[j+'-peak-value'].copy() / temp2['max-cap'].copy()
        temp3['laser'] = abs(temp2[j+'-laser'].copy().astype(float)) # <- abs so that negative steps can be logged later on
        temp3['nerve'] = abs(temp2[j+'-peak-value'].copy().astype(float)) # <- abs so that all values are positive and smaller amplitude CAPs are lower that large amplitude CAPs.
        temp3['step'] = j
        temp3['genotype'] = i
        summary = pd.concat([summary,temp3]).reset_index(drop=True)
for i in 'abcdefghijklmnopqrst':
    temp2 = summary.loc[summary['step']==i]
    summary.loc[temp2.index,'laser-avg'] = abs(temp2['laser'].median()) # <- abs so that negative steps can be logged later on

#

#sb.pointplot(data=summary,x='laser-avg',y='prop-max-cap',hue='genotype',join=False)

#sb.scatterplot(data=summary,x='laser-avg',y='prop-max-cap',hue='genotype')
#mplp.xscale('log')

#sb.stripplot(data=summary,x='laser-avg',y='prop-max-cap',hue='genotype')

sb.lmplot(data=summary,x='laser',y='nerve',hue='genotype',palette=colours)
mplp.xscale('log')

g = sb.lmplot(data=summary,x='laser',y='prop-max-cap',hue='genotype',logistic=True,\
              palette=colours) # order=2
mplp.xscale('log')
g.set_axis_labels('Displacement (nm)','Proportion of max CAP')

sb.lmplot(data=summary,x='laser-avg',y='prop-max-cap',hue='genotype',logistic=True,palette=colours) # order=2
mplp.xscale('log')

order = [i for i in reversed('abcdefghijklmnopqrst')]

fig = mplp.figure()
ax = fig.add_subplot(111)
sb.stripplot(data=summary,x='step',y='nerve',hue='genotype',dodge=True,alpha=0.5,\
             palette=colours,order=order,ax=ax)
sb.pointplot(data=summary,x='step',y='nerve',hue='genotype',join=False,dodge=True,\
             ci=None,estimator=np.median,palette=colours,order=order,ax=ax)
ax.set_xlabel('Force step')
ax.set_ylabel('Nerve (mV)')
h,l = ax.get_legend_handles_labels()
ax.legend(h[:int(len(h)/2)],l[:int(len(l)/2)],loc='upper left',frameon=False)

fig = mplp.figure()
ax = fig.add_subplot(111)
sb.stripplot(data=summary,x='step',y='prop-max-cap',hue='genotype',dodge=True,alpha=0.5,\
             palette=colours,order=order,ax=ax)
sb.pointplot(data=summary,x='step',y='prop-max-cap',hue='genotype',join=False,dodge=True,\
             ci=None,estimator=np.median,palette=colours,order=order,ax=ax)
ax.set_xlabel('Force step')
ax.set_ylabel('Proportion of max CAP')
h,l = ax.get_legend_handles_labels()
ax.legend(h[:int(len(h)/2)],l[:int(len(l)/2)],loc='upper left',frameon=False)

#==============================================================================
#                       B I O P H Y S   A N A L Y S I S
#==============================================================================

# xpeak = initial displacement peak of a flaggelum in response to a force step
# xsteady = steady state displacement of a flaggelum during a force step
# kpeak = model describing force as a function of displacement. This function can be done two ways, one using a simpler equeation (eq1) and one using a more complex equation (eq2).
# ksteady = a single constant describing the linear relationship between force (pN) and displacement at xsteady (um)
# kinfinity = "asymptotic stiffness" of the flaggelum (how much constant stiffness is there in the system i.e. ignoring the impact of channels on stiffness - they are ALL open/closed)

# eq1:
#
# F = Kinf * X - Rho0 (X) * N * z + F0
#
#   F           = force
#   Kinf        = kinfinity
#   X           = displacement distance
#   Rho0 (X)    = the open probability at X, where:
#       X0          = the displacement at which the open probability is one-half
#       kB          = the Boltzmann constant
#       T           = absolute temperature
#   N           = the number of channels
#   z           = the change in force at a single gating spring when the channel opens (does N-channels==N-springs?), seen at the flaggelum
#   F0          = a constant offset term

laser2 = pd.DataFrame(columns=laser[list(laser.keys())[0]].columns)
for i in laser.keys():
    laser2 = pd.concat([laser2,laser[i]]).reset_index(drop=True)
laser2['dtpeak'] = laser2['tpeak'] - laser2['tacce-onset'] # <- delta tpeak (accounts for differences in tacce_onset)
laser2['dtsteady'] = laser2['tsteady'] - laser2['tacce-onset'] # <- delta tpeak
for i in ['xpeak','tpeak','xsteady','tsteady','acce-onset','tacce-onset','dtpeak','dtsteady','steps-zt-bin']:
    laser2[i] = laser2[i].astype(float)

#==============================================================================
#                           C O M B I N A T I O N
#==============================================================================

peaks = peaks.dropna(subset='mosquito-id')
peaks['treatment'] = peaks['mosquito-id'].apply(lambda x: x.split('_')[1] + x[-1])
peaks_ = peaks.dropna(axis=0)

nerve2 = pd.DataFrame(columns=['mosquito-id','treatment','steps-state','step','force','major-cap'])
for i in peaks_['mosquito-id'].unique():
    for j in 'abcdefghijklmnopqrst':
        f_plus = laser[i].loc[laser[i]['step']==j]['force'].values[1]
        f_minus = laser[i].loc[laser[i]['step']==j]['force'].values[0]
        cap_plus = peaks_.loc[(peaks_['mosquito-id']==i) & (peaks_[j+'-stim-onset-id']=='0,1')][j+'-peak-value'].min() # <- CAP of biggest peak
        cap_minus = peaks_.loc[(peaks['mosquito-id']==i) & (peaks_[j+'-stim-onset-id']=='0,-1')][j+'-peak-value'].min() # <- CAP of biggest peak
        gt = peaks_.loc[peaks_['mosquito-id']==i]['treatment'].values[0]
        s = laser[i].loc[laser[i]['step']==j]['steps-state'].values[0]
        temp = pd.DataFrame([[i,gt,s,'-'+j,f_minus,cap_minus],[i,gt,s,j,f_plus,cap_plus]],columns=['mosquito-id','treatment','steps-state','step','force','major-cap'])
        nerve2 = pd.concat([nerve2,temp]).reset_index(drop=True)
for i in ['force','major-cap']:
    nerve2[i] = nerve2[i].astype(float)

nerve3 = pd.DataFrame(columns=['mosquito-id','genotype','steps-state','step','force','major-cap','prop-max-cap'])
for i in nerve2['mosquito-id'].unique():
    temp = nerve2.loc[nerve2['mosquito-id']==i]
    maxcap = temp['major-cap'].min()
    temp['prop-max-cap'] = temp['major-cap'].apply(lambda x: abs(x) / abs(maxcap))
    nerve3 = pd.concat([nerve3,temp]).reset_index(drop=True)

nerve3['treatment'] = nerve3['treatment'].replace({'picrotoxina':'Picrotoxin (1uM) - Baseline', 'picrotoxinb': 'Picrotoxin (1uM)', 'dmsoa': 'Vehicle - Baseline', 'dmsob': 'Vehicle'})

colours = {'Picrotoxin (1uM) - Baseline':'#7fc97f','Picrotoxin (1uM)':'#327632','Vehicle - Baseline':'#beaed4','Vehicle':'#604583'}

_temp = nerve3.dropna(subset=['prop-max-cap', 'force'])

ax = sb.scatterplot(data=_temp,x='force',y='prop-max-cap',hue='treatment',palette=colours)
#ax.set_xscale('log')
ax.legend(title=None)
ax.set_xlim([-20000,20000])
ax.set_ylabel(r'CAP (V/$V_{max}$)')
ax.set_xlabel('Force (fN?)')
