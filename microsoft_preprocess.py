import pandas as pd
import numpy as np, os
import math 

def preprocess_microsoft_malware(directory):
    # Feature Engineering (FE) and One-Hot Encoding (OHE) columns
    FE = ['EngineVersion','AppVersion','AvSigVersion','Census_OSVersion']
    OHE = [ 'RtpStateBitfield','IsSxsPassiveMode','DefaultBrowsersIdentifier',
        'AVProductStatesIdentifier','AVProductsInstalled', 'AVProductsEnabled',
        'CountryIdentifier', 'CityIdentifier', 
        'GeoNameIdentifier', 'LocaleEnglishNameIdentifier',
        'Processor', 'OsBuild', 'OsSuite',
        'SmartScreen','Census_MDC2FormFactor',
        'Census_OEMNameIdentifier', 
        'Census_ProcessorCoreCount',
        'Census_ProcessorModelIdentifier', 
        'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
        'Census_HasOpticalDiskDrive',
        'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches',
        'Census_InternalPrimaryDisplayResolutionHorizontal',
        'Census_InternalPrimaryDisplayResolutionVertical',
        'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
        'Census_InternalBatteryNumberOfCharges',
        'Census_OSEdition', 'Census_OSInstallLanguageIdentifier',
        'Census_GenuineStateName','Census_ActivationChannel',
        'Census_FirmwareManufacturerIdentifier',
        'Census_IsTouchEnabled', 'Census_IsPenCapable',
        'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
        'Wdft_RegionIdentifier']
    # LOAD ALL AS CATEGORIES
    dtypes = {}
    for x in FE+OHE: dtypes[x] = 'category'
    dtypes['MachineIdentifier'] = 'str'
    dtypes['HasDetections'] = 'int8'
    
    # LOAD CSV FILE
    df_train = pd.read_csv(directory+'train.csv', usecols=dtypes.keys(), dtype=dtypes)
    print ('Loaded',len(df_train),'rows of TRAIN.CSV!')

    # DOWNSAMPLE
    sm = 100000
    df_train = df_train.sample(sm)
    print ('Only using',sm,'rows to train and validate')

    cols = []; dd = []

    # ENCODE NEW
    for x in FE:
        cols += encode_FE(df_train,x)
    for x in OHE:
        tmp = encode_OHE(df_train,x,0.005,5)
        cols += tmp[0]; dd.append(tmp[1])
    print('Encoded',len(cols),'new variables')

    # REMOVE OLD
    for x in FE+OHE:
        del df_train[x]
    print('Removed original',len(FE+OHE),'variables')
    
    return df_train



# CHECK FOR NAN
def nan_check(x):
    if isinstance(x,float):
        if math.isnan(x):
            return True
    return False

# FREQUENCY ENCODING
def encode_FE(df,col,verbose=1):
    d = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(d)/d.max()
    if verbose==1:
        print('FE encoded',col)
    return [n]

# ONE-HOT-ENCODE ALL CATEGORY VALUES THAT COMPRISE MORE THAN
# "FILTER" PERCENT OF TOTAL DATA AND HAS SIGNIFICANCE GREATER THAN "ZVALUE"
def encode_OHE(df, col, filter, zvalue, tar='HasDetections', m=0.5, verbose=1):
    cv = df[col].value_counts(dropna=False)
    cvd = cv.to_dict()
    vals = len(cv)
    th = filter * len(df)
    sd = zvalue * 0.5/ math.sqrt(th)
    #print(sd)
    n = []; ct = 0; d = {}
    for x in cv.index:
        try:
            if cv[x]<th: break
            sd = zvalue * 0.5/ math.sqrt(cv[x])
        except:
            if cvd[x]<th: break
            sd = zvalue * 0.5/ math.sqrt(cvd[x])
        if nan_check(x): r = df[df[col].isna()][tar].mean()
        else: r = df[df[col]==x][tar].mean()
        if abs(r-m)>sd:
            nm = col+'_BE_'+str(x)
            if nan_check(x): df[nm] = (df[col].isna()).astype('int8')
            else: df[nm] = (df[col]==x).astype('int8')
            n.append(nm)
            d[x] = 1
        ct += 1
        if (ct+1)>=vals: break
    if verbose==1:
        print('OHE encoded',col,'- Created',len(d),'booleans')
    return [n,d]

# ONE-HOT-ENCODING from dictionary
def encode_OHE_test(df,col,dt):
    n = []
    for x in dt: 
        n += encode_BE(df,col,x)
    return n

# BOOLEAN ENCODING
def encode_BE(df,col,val):
    n = col+"_BE_"+str(val)
    if nan_check(val):
        df[n] = df[col].isna()
    else:
        df[n] = df[col]==val
    df[n] = df[n].astype('int8')
    return [n]
