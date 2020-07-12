#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:02:53 2019

@author: omar

Orden de las carpetas:

    |02_4KHZ_4096                    
        |.1185619968
        |.1185669120
        |.1185673216
        |.1185677312
            |..H1_1185677312.hdf5
            |..L1_1185677312.hdf5
            |..V1_1185677312.hdf5
        |.1185681408
        |.1185718272
        |.1185755136
        |.1185759232
        |.1185767424
        |.1185771520
        
    |signal_data
        |.signal_e15a_ls.dat
        |.signal_e15a_shen.dat
        ...
        |.signal_s40a3o15_shen.dat
"""
#%% Imports

import numpy as np
from numpy import genfromtxt
import pandas as pd
from pandas import read_csv
import h5py
import os 
import re
import random
import copy
import matplotlib.pyplot as plt

#%% readligo

def read_frame(filename, ifo, readstrain=True, strain_chan=None, dq_chan=None, inj_chan=None):
    """
    Helper function to read frame files
    """

    from gwpy.timeseries import TimeSeries


    if ifo is None:
        raise TypeError("""To read GWF data, ifo must be 'H1', 'H2', or 'L1'.
        def loaddata(filename, ifo=None):""")

    #-- Read strain channel
    if strain_chan is None:
        strain_chan = ifo + ':LOSC-STRAIN'
    
    if readstrain:
        try:
            sd = TimeSeries.read(filename, strain_chan)
            strain = sd.value
            gpsStart = sd.t0.value
            ts = sd.dt.value
        except:
            print("ERROR reading file {0} with strain channel {1}".format(filename, strain_chan))
            raise
    else:
        ts = 1
        strain = 0
    
    #-- Read DQ channel
    if dq_chan is None:
        dq_chan = ifo + ':LOSC-DQMASK'

    try:
        qd = TimeSeries.read(str(filename), str(dq_chan))
        gpsStart = qd.t0.value
        qmask = np.array(qd.value)
        dq_ts = qd.dt.value
        shortnameList_wbit = str(qd.unit).split()
        shortnameList = [name.split(':')[1] for name in shortnameList_wbit]
    except:
        print("ERROR reading DQ channel '{0}' from file: {1}".format(dq_chan, filename))
        raise

    #-- Read Injection channel
    if inj_chan is None:
        inj_chan = ifo + ':LOSC-INJMASK'
    
    try:
        injdata = TimeSeries.read(str(filename), str(inj_chan))
        injmask = injdata.value
        injnamelist_bit = str(injdata.unit).split()
        injnamelist     = [name.split(':')[1] for name in injnamelist_bit]
    except:
        print("ERROR reading injection channel '{0}' from file: {1}".format(inj_chan, filename))
        raise

    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnamelist
    
def read_hdf5(filename, readstrain=True):
    """
    Helper function to read HDF5 files
    """
    import h5py
    dataFile = h5py.File(filename, 'r')

    #-- Read the strain
    if readstrain:
        strain = dataFile['strain']['Strain'][...]
    else:
        strain = 0

    ts = dataFile['strain']['Strain'].attrs['Xspacing']
    
    #-- Read the DQ information
    dqInfo = dataFile['quality']['simple']
    qmask = dqInfo['DQmask'][...]
    shortnameArray = dqInfo['DQShortnames'].value
    shortnameList  = list(shortnameArray)
    
    # -- Read the INJ information
    injInfo = dataFile['quality/injections']
    injmask = injInfo['Injmask'][...]
    injnameArray = injInfo['InjShortnames'].value
    injnameList  = list(injnameArray)
    
    #-- Read the meta data
    meta = dataFile['meta']
    gpsStart = meta['GPSstart'].value    
    
    dataFile.close()
    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList


def loaddata(filename, ifo=None, tvec=True, readstrain=True, strain_chan=None, dq_chan=None, inj_chan=None):
    """
    The input filename should be a LOSC .hdf5 file or a LOSC .gwf
    file.  The file type will be determined from the extenstion.  
    The detector should be H1, H2, or L1.

    The return value is: 
    STRAIN, TIME, CHANNEL_DICT

    STRAIN is a vector of strain values
    TIME is a vector of time values to match the STRAIN vector
         unless the flag tvec=False.  In that case, TIME is a
         dictionary of meta values.
    CHANNEL_DICT is a dictionary of data quality channels    
    STRAIN_CHAN is the channel name of the strain vector in GWF files.
    DQ_CHAN is the channel name of the data quality vector in GWF files.
    INJ_CHAN is the channel name of the injection vector in GWF files.
    """

    # -- Check for zero length file
    try:
        if os.stat(filename).st_size == 0:
            return None, None, None
    except:
        return None,None,None

    file_ext = os.path.splitext(filename)[1]    
    if (file_ext.upper() == '.GWF'):
        strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = read_frame(filename, ifo, readstrain, strain_chan, dq_chan, inj_chan)
    else:
        strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = read_hdf5(filename, readstrain)
        
    #-- Create the time vector
    gpsEnd = gpsStart + len(qmask)
    if tvec:
        time = np.arange(gpsStart, gpsEnd, ts)
    else:
        meta = {}
        meta['start'] = gpsStart
        meta['stop']  = gpsEnd
        meta['dt']    = ts

    #-- Create 1 Hz DQ channel for each DQ and INJ channel
    channel_dict = {}  #-- 1 Hz, mask
    slice_dict   = {}  #-- sampling freq. of stain, a list of slices
    final_one_hz = np.zeros(qmask.shape, dtype='int32')
    for flag in shortnameList:
        bit = shortnameList.index(flag)
        # Special check for python 3
        if isinstance(flag, bytes): flag = flag.decode("utf-8") 
        
        channel_dict[flag] = (qmask >> bit) & 1

    for flag in injnameList:
        bit = injnameList.index(flag)
        # Special check for python 3
        if isinstance(flag, bytes): flag = flag.decode("utf-8") 
        
        channel_dict[flag] = (injmask >> bit) & 1
       
    #-- Calculate the DEFAULT channel
    try:
        channel_dict['DEFAULT'] = ( channel_dict['DATA'] )
    except:
        print("Warning: Failed to calculate DEFAULT data quality channel")

    if tvec:
        return strain, time, channel_dict
    else:
        return strain, meta, channel_dict

#%% resample

def resample(raw_template, RAW_TEMPLATE_DIR, distance):
    
    # Raw template 
    time_gw  = np.loadtxt(os.path.join(RAW_TEMPLATE_DIR, raw_template),usecols=(0)) # ms
    strain_gw= np.loadtxt(os.path.join(RAW_TEMPLATE_DIR, raw_template),usecols=(1)) # 10 Kpc
    time_gw  = time_gw/1000 # s             # dim(8192,)
    strain_gw= strain_gw/(distance/10)      # {distance} Kpc  
    
    # Hdf5 to csv
    cols = np.empty((len(time_gw), 2))
    cols[:, 0] = time_gw
    cols[:, 1] = strain_gw
    c = np.savetxt('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), cols, delimiter =' ', header = "Time, Strain", comments="")
    gw = read_csv('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), delim_whitespace=True)
    gw.columns = ['Time', 'Strain']
    gw.to_csv('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), sep = ' ', index = False)
    
    # Resample
    gw = read_csv('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), delim_whitespace=True, index_col = 0)
    gw.index = pd.to_datetime(gw.index, unit='s')
    downsampled = gw.resample('244140ns').ffill()
    conversion = 244140/1000000000 #-- ts_ns 
    sampled_strain_gw= downsampled['Strain'].to_numpy()
    sampled_time_gw  = np.arange(0, conversion * len(sampled_strain_gw), conversion)
    
    # Generate text file resampled
    cols = np.empty((len(sampled_time_gw), 2))
    cols[:, 0] = sampled_time_gw
    cols[:, 1] = sampled_strain_gw
    c = np.savetxt('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), cols, delimiter =' ', header = "Time, Strain", comments="")
    gw = read_csv('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), delim_whitespace=True)
    gw.columns = ['Time', 'Strain']
    gw.to_csv('/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template), sep = ' ', index = False)

#%% injection
                       
distances = [1,10,20,30,40,50,60,70,80,90,100] #[kpc]
times = [30,32,55,81,100,173,500]              #[s]
variations = [2,4,7,15,16,21,27]               #[s]
detectors = ['H1','L1','V1']
RAW_TEMPLATE_DIR ='/home/omar/GW/signal_data'
GPS_TIME_DIR ='/home/omar/GW/02_4KHZ_4096' 
TEMPLATE_DIR ='/home/omar/GW/resampled'

for gps in os.listdir(GPS_TIME_DIR):
    # Read LIGO (noise) 
    DETECTOR_DIR ='/home/omar/GW/02_4KHZ_4096/{}'.format(gps) 
    H1= DETECTOR_DIR +'/'+ detectors[0] +'_'+ gps +'.hdf5'
    L1= DETECTOR_DIR +'/'+ detectors[1] +'_'+ gps +'.hdf5'
    V1= DETECTOR_DIR +'/'+ detectors[2] +'_'+ gps +'.hdf5'
    
    strain_H1, time_H1, channel_dict_H1 = loaddata(H1)
    strain_L1, time_L1, channel_dict_L1 = loaddata(L1)
    strain_V1, time_V1, channel_dict_V1 = loaddata(V1)
    
    dataFile_H1= h5py.File(H1, 'r')
    ts = dataFile_H1['strain']['Strain'].attrs['Xspacing'] # time sample ruido (0.000244140625 s)
    
    # Arrays for injections
    strain_H1_injected = copy.deepcopy(strain_H1)
    strain_L1_injected = copy.deepcopy(strain_L1)
    strain_V1_injected = copy.deepcopy(strain_V1)
    
    # Arrays for logs
    log_start_H1 = [] # Inicio de la inyección para H1 
    log_end_H1   = [] # Fin de la inyección para H1 
    log_start_L1 = [] # Inicio de la inyección para L1 (desfase) 
    log_end_L1   = [] # Fin de la inyección para L1 (desfase)
    log_start_V1 = [] # Inicio de la inyección para V1 (desfase) 
    log_end_V1   = [] # Fin de la inyección para V1 (desfase)
    log_dis      = [] # Distancia de la fuente
    log_presup   = [] # Fuente inyectada
    log_ede      = [] # Ec. de Edo. de la fuente
    
    # Variables para las ventanas de inyeccion
    start = random.randint(0,int(4/ts))  # Iniciar a inyectar entre el segundo 0 y 4
    end   = len(time_H1)                 # La duración de los detectores es la misma
    dt    = int(random.choice(times)/ts) # 32 s 
    
    # Ciclo para inyectar el template a los tres ruidos
    for i in range(start,end+1,dt):
        # Elegir un template y una distancia al azar del catalogo
        raw_template = random.choice(os.listdir(RAW_TEMPLATE_DIR))
        distance = random.choice(distances)
        
        # Resample
        resample(raw_template, RAW_TEMPLATE_DIR, distance)
        
        # Leer template resampleado (y quitar encabezados)
        temp_dir=  '/home/omar/GW/resampled/{}kpc-{}.csv'.format(distance,raw_template)
        template= genfromtxt(temp_dir, delimiter=' ')
        template= np.delete(template,0,0)
        
        # Tipo de presupernova para log
        presupernovae = re.search('-signal_(.*)_',temp_dir)
        presupernovae = presupernovae.group(1)
        
        # Ecuacion de estado para log
        ede = re.search(presupernovae+'_(.*).dat',temp_dir)
        ede = ede.group(1)
        
        # Para inyectar dentro de la duración del ruido
        if i+len(template[:,0]) <= (end+1)-len(template[:,0]):
            # Desfase de 10 ms para L1 respecto H1
            gapL1 = i+int(10/(1000*ts))
            # Desfase de 14 ms para V1 respecto H1
            gapV1 = i+int(14/(1000*ts))
            
            # Injección
            strain_H1_injected[i:i+len(template[:,0])] += template[:,1]
            strain_L1_injected[gapL1:gapL1+len(template[:,0])] += template[:,1]
            strain_V1_injected[gapV1:gapV1+len(template[:,0])] += template[:,1]
            
            log_start_H1.append(i)
            log_end_H1.append(i+len(template[:,0]))
            log_start_L1.append(gapL1)
            log_end_L1.append(gapL1+len(template[:,0]))     
            log_start_V1.append(gapV1)
            log_end_V1.append(gapV1+len(template[:,0]))               
            log_dis.append(distance)
            log_presup.append(presupernovae)
            log_ede.append(ede)
                
        variation = random.randint(-int(random.choice(variations)/ts),int(random.choice(variations)/ts)) 
        dt = int(random.choice(times)/ts) + variation #?s +- ¿s
            
    # Guardar ruido inyectado 
    inj_H1 = np.empty((len(strain_H1_injected),2))
    inj_H1[:,0] = time_H1
    inj_H1[:,1] = strain_H1_injected
    Hi  = np.savetxt('/home/omar/GW/dataset/data/H1r_{}.csv'.format(gps), inj_H1, delimiter =' ', header = 'Time, Strain', comments="")
    H_inj = read_csv('/home/omar/GW/dataset/data/H1r_{}.csv'.format(gps), delim_whitespace=True)
    H_inj.columns = ['Time', 'Strain']
    H_inj.to_csv('/home/omar/GW/dataset/data/H1r_{}.csv'.format(gps), sep = ' ', index = False)
    
    inj_L1 = np.empty((len(strain_L1_injected),2))
    inj_L1[:,0] = time_L1
    inj_L1[:,1] = strain_L1_injected
    Li  = np.savetxt('/home/omar/GW/dataset/data/L1r_{}.csv'.format(gps), inj_L1, delimiter =' ', header = 'Time, Strain', comments="")
    L_inj = read_csv('/home/omar/GW/dataset/data/L1r_{}.csv'.format(gps), delim_whitespace=True)
    L_inj.columns = ['Time', 'Strain']
    L_inj.to_csv('/home/omar/GW/dataset/data/L1r_{}.csv'.format(gps), sep = ' ', index = False)
    
    inj_V1 = np.empty((len(strain_V1_injected),2))
    inj_V1[:,0] = time_V1
    inj_V1[:,1] = strain_V1_injected
    Vi  = np.savetxt('/home/omar/GW/dataset/data/V1r_{}.csv'.format(gps), inj_V1, delimiter =' ', header = 'Time, Strain', comments="")
    V_inj = read_csv('/home/omar/GW/dataset/data/V1r_{}.csv'.format(gps), delim_whitespace=True)
    V_inj.columns = ['Time', 'Strain']
    V_inj.to_csv('/home/omar/GW/dataset/data/V1r_{}.csv'.format(gps), sep = ' ', index = False)
        
    # Crear un archivo de texto para logs   
    data_H1 = {'Start_Inj':log_start_H1,
            'End_Inj':log_end_H1,
            'Distance':log_dis,
            'Presupernovae':log_presup,
            'Ede':log_ede}
    logs_H1 = pd.DataFrame(data_H1)
        
    data_L1 = {'Start_Inj':log_start_L1,
               'End_Inj':log_end_L1,
               'Distance':log_dis,
               'Presupernovae':log_presup,
               'Ede':log_ede}
    logs_L1 = pd.DataFrame(data_L1)
    
    data_V1 = {'Start_Inj':log_start_V1,
               'End_Inj':log_end_V1,
               'Distance':log_dis,
               'Presupernovae':log_presup,
               'Ede':log_ede}
    logs_V1 = pd.DataFrame(data_V1)
    
    for detector in detectors:  
        if detector == 'H1':
           logs_H1.to_csv('/home/omar/GW/dataset/logs/{}r_{}.csv'.format(detector,gps), sep = ' ', index = False)
        if detector == 'L1':
           logs_L1.to_csv('/home/omar/GW/dataset/logs/{}r_{}.csv'.format(detector,gps), sep = ' ', index = False)
        if detector == 'V1':
           logs_V1.to_csv('/home/omar/GW/dataset/logs/{}r_{}.csv'.format(detector,gps), sep = ' ', index = False)
#%%  Para comprobar  
inj    = 0
inicio = log_start_H1[inj]
fin    = log_end_H1[inj]
print(strain_H1[inicio], '\n', strain_H1_injected[inicio])
#%% Plot

inicio   = log_start_H1[0]
fin      = log_end_H1[0]

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)

#plt.figure("SOFTWARE INJECTION H1") #zoom
ax.plot(time_H1[inicio:fin], strain_H1_injected[inicio:fin], linewidth=1.0,c='orange', label='H1 w/ SW-INJ')
ax.plot(time_H1[inicio:fin], strain_H1[inicio:fin], linewidth=1.0,c='g', label='H1 wo/ SW-INJ')
#ax.set_xlabel('Time (s)')
#ax.set_ylabel('Strain')
#ax.legend(loc='lower right')
#plt.show()    

#plt.figure("SOFTWARE INJECTION L1") #zoom
ax.plot(time_L1[inicio:fin], strain_L1_injected[inicio:fin], linewidth=1.0,c='orange', label='L1 w/ SW-INJ')
ax.plot(time_L1[inicio:fin], strain_L1[inicio:fin], linewidth=1.0, c='g',label='L1 wo/ SW-INJ')
#ax.set_xlabel('Time (s)')
#ax.set_ylabel('Strain')
#ax.legend(loc='lower right')
#plt.show()

#plt.figure("SOFTWARE INJECTION V1") #zoom
ax.plot(time_V1[inicio:fin], strain_V1_injected[inicio:fin], linewidth=1.0,c='orange', label='L1 w/ SW-INJ')
ax.plot(time_V1[inicio:fin], strain_V1[inicio:fin], linewidth=1.0, c='g', label='L1 wo/ SW-INJ')
ax.plot([inicio,-1.5],[inicio,1.5],'k-',alpha=0.2,lw=1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.legend(loc='lower right')
plt.show()
