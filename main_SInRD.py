import pandas as pd
import numpy as np
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import date
from datetime import timedelta as td, datetime as datetime
from scipy.optimize import curve_fit
import warnings
import sys
#=====================================================
M = 500;
lowper = 0.25;
mean = .50;
highper = .75;
infer_datetime_format=True
print("1")
#=====================================================
#functions organize data
def time_cal(dt_fulldata):
    T = len(dt_fulldata);
    initialdatadate = dt_fulldata.index.tolist()[0];
    sdate1 = initialdatadate.date();
    edate1 = sdate1 + td(days = T + 1)
    pastplot_t = pd.date_range(sdate1,edate1-td(days=1),freq='d')
    sdate2 = edate1 + td(days = 1)
    edate2 = sdate2 + td(days = T + 1)
    futplot_t = pd.date_range(sdate2,edate2-td(days=1),freq='d')
    fullplot_t = pd.date_range(sdate1,edate2-td(days=1),freq='d')
    return pastplot_t, futplot_t,fullplot_t,sdate2
def log_func(x, a, b):
    invalid_mask = (x <= 0)
    result = np.zeros_like(x)
    result[~invalid_mask] = a * np.log(x[~invalid_mask]) + b

    return result

st.set_page_config(page_title="SInRD GUI V5", layout="wide")
@st.cache_data
def load_data():
    def process_country_data(country_code, population, data_prefix, parameter_file):
        sim_S = pd.read_csv(f'data/{data_prefix}_full_sim_S.csv').transpose()
        sim_I1 = pd.read_csv(f'data/{data_prefix}_full_sim_I1.csv').transpose()
        sim_I2 = pd.read_csv(f'data/{data_prefix}_full_sim_I2.csv').transpose() if pd.io.common.file_exists(f'data/{data_prefix}_full_sim_I2.csv') else None
        sim_I3 = pd.read_csv(f'data/{data_prefix}_full_sim_I3.csv').transpose() if pd.io.common.file_exists(f'data/{data_prefix}_full_sim_I3.csv') else None
        sim_R = pd.read_csv(f'data/{data_prefix}_full_sim_R.csv').transpose()
        sim_D = pd.read_csv(f'data/{data_prefix}_full_sim_D.csv').transpose() if pd.io.common.file_exists(f'data/{data_prefix}_full_sim_D.csv') else None

        dt_fulldata = pd.read_csv(f'data/si3rd_data_{country_code}.csv') 
        dt_fulldata['date'] = pd.to_datetime(dt_fulldata['date'])
        dt_fulldata = dt_fulldata.set_index(['date'])
        dt_fulldata = dt_fulldata.tail(sim_S.shape[0])

        dt_mcmcresult = pd.read_csv(parameter_file)
        dt_mcmcresult = dt_mcmcresult.dropna()
        dt_mcmcresult = dt_mcmcresult.tail(sim_S.shape[0])
        mcmcsize = np.size(dt_mcmcresult, axis=0)

        # Calculate time-related variables
        pastplot, futplot, fullplot, sdate2 = time_cal(dt_fulldata)
        N = len(sim_S)

        # Initialize percentile arrays
        percentile_S = np.ones((N, 3))
        percentile_I1 = np.ones((N, 3))
        percentile_I2 = np.ones((N, 3))
        percentile_I3 = np.ones((N, 3))
        percentile_R = np.ones((N, 3))
        percentile_D = np.ones((N, 3))

        for j in np.linspace(0, len(sim_S) - 1, len(sim_S)):
            percentile_S[j.astype(int)] = (pd.Series(sim_S.iloc[j.astype(int), :])).quantile([lowper, mean, highper])
            percentile_I1[j.astype(int)] = (pd.Series(sim_I1.iloc[j.astype(int), :])).quantile([lowper, mean, highper])
            percentile_I2[j.astype(int)] = (lambda x: pd.Series(x).quantile([lowper, mean, highper]) if x is not None else None)(sim_I2.iloc[j.astype(int), :]) if sim_I2 is not None else None
            percentile_I3[j.astype(int)] = (lambda x: pd.Series(x).quantile([lowper, mean, highper]) if x is not None else None)(sim_I3.iloc[j.astype(int), :]) if sim_I3 is not None else None
            percentile_R[j.astype(int)] = (pd.Series(sim_R.iloc[j.astype(int), :])).quantile([lowper, mean, highper])
            percentile_D[j.astype(int)] = (lambda x: pd.Series(x).quantile([lowper, mean, highper]) if x is not None else None)(sim_D.iloc[j.astype(int), :]) if sim_D is not None else None

        def fit_params(param_name):
            try:
                pars = curve_fit(f=log_func, xdata=np.linspace(1, N, N), ydata=dt_mcmcresult[param_name], p0=[0, 0], bounds=(-np.inf, np.inf))[0]
            except:
                try:
                    pars = curve_fit(f=log_func, xdata=np.linspace(1, N, N), ydata=dt_mcmcresult[param_name[:1]], p0=[0, 0], bounds=(-np.inf, np.inf))[0]
                except:
                    pars = 0 
            return pars
        
        pars_r1 = fit_params('r1')
        pars_r2 = fit_params('r2')
        pars_r3 = fit_params('r3')
        pars_d1 = fit_params('d1')
        pars_d2 = fit_params('d2')
        pars_d3 = fit_params('d3')
        pars_p1 = fit_params('p1')
        pars_p2 = fit_params('p2')
        pars_p3 = fit_params('p3')

        return {
            'percentile_S': percentile_S,
            'percentile_I1': percentile_I1,
            'percentile_I2': percentile_I2,
            'percentile_I3': percentile_I3,
            'percentile_R': percentile_R,
            'percentile_D': percentile_D,
            'pars_r1': pars_r1,
            'pars_r2': pars_r2,
            'pars_r3': pars_r3,
            'pars_d1': pars_d1, 
            'pars_d2': pars_d2,
            'pars_d3': pars_d3,
            'pars_p1': pars_p1,
            'pars_p2': pars_p2,
            'pars_p3': pars_p3,
            'pastplot': pastplot,
            'futplot': futplot,
            'fullplot': fullplot,
            'sdate2': sdate2,
            'fulldata':dt_fulldata,
            'mcmcresult':dt_mcmcresult
        }
    sircountries = {
        'Malaysia': ('Malaysia', 3.153*10**7, 'SIR_Malaysia', 'data/Malaysia-SIR-param-estimate.csv'),
        'Canada': ('Canada', 38.25*10**6, 'SIR_Canada', 'data/Canada-SIR-param-estimate.csv'),
        'Italy': ('Italy', 59.11*10**6, 'SIR_Italy', 'data/Italy-SIR-param-estimate.csv'),
        'Slovakia': ('Slovakia', 38.25*10**6, 'SIR_Slovakia', 'data/Slovakia-SIR-param-estimate.csv'),
        'SouthAfrica': ('SouthAfrica', 38.25*10**6, 'SIR_SouthAfrica', 'data/South-Africa-SIR-param-estimate.csv')
    }
    sirdcountries = {
        'Malaysia': ('Malaysia', 3.153*10**7, 'SIRD_Malaysia', 'data/Malaysia-SIRD-param-estimate.csv'),
        'Canada': ('Canada', 38.25*10**6, 'SIRD_Canada', 'data/Canada-SIRD-param-estimate.csv'),
        'Italy': ('Italy', 59.11*10**6, 'SIRD_Italy', 'data/Italy-SIRD-param-estimate.csv'),
        'Slovakia': ('Slovakia', 38.25*10**6, 'SIRD_Slovakia', 'data/Slovakia-SIRD-param-estimate.csv'),
        'SouthAfrica': ('SouthAfrica', 38.25*10**6, 'SIRD_SouthAfrica', 'data/South-Africa-SIRD-param-estimate.csv')
    }
    lsirdcountries = {
        'Malaysia': ('Malaysia', 3.153*10**7, 'L_SIRD_Malaysia', 'data/Malaysia-l-SIRD-param-estimate.csv'),
        'Canada': ('Canada', 38.25*10**6, 'L_SIRD_Canada', 'data/Canada-l-SIRD-param-estimate.csv'),
        'Italy': ('Italy', 59.11*10**6, 'L_SIRD_Italy', 'data/Italy-l-SIRD-param-estimate.csv'),
        'Slovakia': ('Slovakia', 38.25*10**6, 'L_SIRD_Slovakia', 'data/Slovakia-l-SIRD-param-estimate.csv'),
        'SouthAfrica': ('SouthAfrica', 38.25*10**6, 'L_SIRD_SouthAfrica', 'data/South-Africa-l-SIRD-param-estimate.csv')
    }
    si3rdcountries = {
        'Malaysia': ('Malaysia', 3.153*10**7, 'SI3RD_Malaysia', 'data/Malaysia-SI3RD-param-estimate.csv'),
        'Canada': ('Canada', 38.25*10**6, 'SI3RD_Canada', 'data/Canada-SI3RD-param-estimate.csv'),
        'Italy': ('Italy', 59.11*10**6, 'SI3RD_Italy', 'data/Italy-SI3RD-param-estimate.csv'),
        'Slovakia': ('Slovakia', 38.25*10**6, 'SI3RD_Slovakia', 'data/Slovakia-SI3RD-param-estimate.csv'),
        'SouthAfrica': ('SouthAfrica', 38.25*10**6, 'SI3RD_SouthAfrica', 'data/South-Africa-SI3RD-param-estimate.csv')
    }
    lsi3rdcountries = {
        'Malaysia': ('Malaysia', 3.153*10**7, 'L_SI3RD_Malaysia', 'data/Malaysia-l-SI3RD-param-estimate.csv'),
        'Canada': ('Canada', 38.25*10**6, 'L_SI3RD_Canada', 'data/Canada-l-SI3RD-param-estimate.csv'),
        'Italy': ('Italy', 59.11*10**6, 'L_SI3RD_Italy', 'data/Italy-l-SI3RD-param-estimate.csv'),
        'Slovakia': ('Slovakia', 38.25*10**6, 'L_SI3RD_Slovakia', 'data/Slovakia-l-SI3RD-param-estimate.csv'),
        'SouthAfrica': ('SouthAfrica', 38.25*10**6, 'L_SI3RD_SouthAfrica', 'data/South-Africa-l-SI3RD-param-estimate.csv')
    }
    
    results_sir = {sircountries: process_country_data(*params) for sircountries, params in sircountries.items()}
    results_sird = {sirdcountries: process_country_data(*params) for sirdcountries, params in sirdcountries.items()}
    results_lsird = {lsirdcountries: process_country_data(*params) for lsirdcountries, params in lsirdcountries.items()}
    results_si3rd = {si3rdcountries: process_country_data(*params) for si3rdcountries, params in si3rdcountries.items()}
    results_lsi3rd = {lsi3rdcountries: process_country_data(*params) for lsi3rdcountries, params in lsi3rdcountries.items()}
    
    results = {"SIR": results_sir,
               "SIRD": results_sird,
               "l-SIRD": results_lsird,
               "SI3RD": results_si3rd,
               "l-SI3RD": results_lsi3rd
               }
    return results

data_SI3RD = load_data();
#=====================================================
#=====================================================
#SRK-4 Tableus form
A = np.array([[0  ,0  ,0,0],
              [1/2,0  ,0,0],
              [0  ,1/2,0,0],
              [0  ,0  ,1,0]]);
B_1 = np.array([[0            ,0            ,0          ,0],
                [-0.7242916356,0            ,0          ,0],
                [0.4237353406 ,-0.1994437050,0          ,0],
                [-1.578475506 ,0.840100343  ,1.738375163,0]]); 
B_2 = np.array([[0           ,0,0,0],
                [2.702000410 ,0,0,0],
                [1.757261649 ,0,0,0],
                [-2.918524118,0,0,0]]); 
G_1 = np.array([[-.7800788474],
                [0.07363768240],
                [1.486520013],
                [0.2199211524]]); 
G_2 = np.array([[1.693950844],
                [1.636107882],
                [-3.024009558],
                [-0.3060491602]]); 
# alpha = np.array([1/6, 1/3, 1/3, 1/6]);
#=====================================================
def SI3RD_func(t0,dt,N,M,Szero,I1zero,I2zero,I3zero,Rzero,Dzero,Ptotal,radio_option, toggle_button_index,date_inputs,strictness_inputs):
    #=====================================================
    fS  = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:mu*(S+I1+I2+I3+R)-S/Ptotal*(Beta1*I1+Beta2*I2+Beta3*I3) + gamma*R - mu*S   
    gS  = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:-sigma*I1*S/Ptotal
    #=====================================================
    fI1 = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:p1*S/Ptotal*(Beta1*I1+Beta2*I2+Beta3*I3)-(r1+d1+mu)*I1
    gI1 = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:sigma*p1*I1*S/Ptotal
    #=====================================================
    fI2 = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:p2*S/Ptotal*(Beta1*I1+Beta2*I2+Beta3*I3)-(r2+d2+mu)*I2
    gI2 = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:sigma*p2*I1*S/Ptotal
    #=====================================================
    fI3 = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:p3*S/Ptotal*(Beta1*I1+Beta2*I2+Beta3*I3)-(r3+d3+mu)*I3
    gI3 = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:sigma*p3*I1*S/Ptotal
    #=====================================================
    fD  = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:d1*I1+d2*I2+d3*I3
    gD  = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:0
    #=====================================================
    fR  = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:r1*I1+r2*I2+r3*I3 - gamma*R - mu*R
    gR  = lambda S,I1,I2,I3,R,D,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu:0
    #=====================================================
        
    SdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    SdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    I1dW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    I1dW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    I2dW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    I2dW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    I3dW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    I3dW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    RdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    RdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    DdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    DdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
        
    Stemp = Szero*np.ones((1,M+1));
    I1temp = I1zero*np.ones((1,M+1));
    I2temp = I2zero*np.ones((1,M+1));
    I3temp = I3zero*np.ones((1,M+1));
    Rtemp = Rzero*np.ones((1,M+1));
    Dtemp = Dzero*np.ones((1,M+1));
    
    Ssrk4_2 = np.ones((N+1,M+1));
    I1srk4_2 = np.ones((N+1,M+1));
    I2srk4_2 = np.ones((N+1,M+1));
    I3srk4_2 = np.ones((N+1,M+1));
    Rsrk4_2 = np.ones((N+1,M+1));
    Dsrk4_2 = np.ones((N+1,M+1));
    #=====================================================
        
    for j in np.linspace(t0,t0+N,N+1):
        Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu = parameter_func(j,radio_option, date_inputs,strictness_inputs);
        SWinc1 = SdW1[int(j) - t0];
        I1Winc1 = I1dW1[int(j) - t0];
        I2Winc1 = I2dW1[int(j) - t0];
        I3Winc1 = I3dW1[int(j) - t0];
        RWinc1 = RdW1[int(j) - t0];
        DWinc1 = DdW1[int(j) - t0];
        SWinc2 = SdW2[int(j) - t0];
        I1Winc2 = I1dW2[int(j) - t0];
        I2Winc2 = I2dW2[int(j) - t0];
        I3Winc2 = I3dW2[int(j) - t0];
        RWinc2 = RdW2[int(j) - t0];
        DWinc2 = DdW2[int(j) - t0];
        
        SJ10 = (0.5*(dt**(1/2))*(SWinc1+(1/np.sqrt(3))*SWinc2));
        I1J10 = (0.5*(dt**(1/2))*(I1Winc1+(1/np.sqrt(3))*I1Winc2));
        I2J10 = (0.5*(dt**(1/2))*(I2Winc1+(1/np.sqrt(3))*I2Winc2));
        I3J10 = (0.5*(dt**(1/2))*(I3Winc1+(1/np.sqrt(3))*I3Winc2));
        RJ10 = (0.5*(dt**(1/2))*(RWinc1+(1/np.sqrt(3))*RWinc2));
        DJ10 = (0.5*(dt**(1/2))*(DWinc1+(1/np.sqrt(3))*DWinc2));
        
        S1 = Stemp;
        I11 = I1temp;
        I21 = I2temp;
        I31 = I3temp;
        R1 = Rtemp;
        D1 = Rtemp;
        
        f1_S = fS(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g1_S = gS(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f1_I1 = fI1(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g1_I1 = gI1(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f1_I2 = fI2(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g1_I2 = gI2(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f1_I3 = fI3(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g1_I3 = gI3(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f1_R = fR(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g1_R = gR(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f1_D = fD(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g1_D = gD(S1,I11,I21,I31,R1,D1,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        
        S2 = Stemp + dt*(A[0,1]*f1_S) + (B_1[0,1]*SWinc1 + B_2[0,1]*SJ10)*g1_S;
        I12 = I1temp + dt*(A[0,1]*f1_I1) + (B_1[0,1]*I1Winc1 + B_2[0,1]*I1J10)*g1_I1;
        I22 = I2temp + dt*(A[0,1]*f1_I2) + (B_1[0,1]*I2Winc1 + B_2[0,1]*I2J10)*g1_I2;
        I32 = I3temp + dt*(A[0,1]*f1_I3) + (B_1[0,1]*I3Winc1 + B_2[0,1]*I3J10)*g1_I3;
        R2 = Rtemp + dt*(A[0,1]*f1_R) + (B_1[0,1]*RWinc1 + B_2[0,1]*RJ10)*g1_R;
        D2 = Dtemp + dt*(A[0,1]*f1_D) + (B_1[0,1]*DWinc1 + B_2[0,1]*DJ10)*g1_D;
        
        f2_S = fS(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g2_S = gS(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f2_I1 = fI1(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g2_I1 = gI1(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f2_I2 = fI2(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g2_I2 = gI2(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f2_I3 = fI3(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g2_I3 = gI3(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f2_R = fR(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g2_R = gR(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f2_D = fD(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g2_D = gD(S2,I12,I22,I32,R2,D2,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        
        S3 = Stemp + dt*(A[1,2]*f2_S) + (B_1[0,2]*SWinc1 + B_2[0,2]*SJ10)*g2_S + (B_1[1,2]*SJ10*g2_S);
        I13 = I1temp + dt*(A[1,2]*f2_I1) + (B_1[0,2]*I1Winc1 + B_2[0,2]*I1J10)*g2_I1 + (B_1[1,2]*I1J10*g2_I1);
        I23 = I2temp + dt*(A[1,2]*f2_I2) + (B_1[0,2]*I2Winc1 + B_2[0,2]*I2J10)*g2_I2 + (B_1[1,2]*I2J10*g2_I2);
        I33 = I3temp + dt*(A[1,2]*f2_I3) + (B_1[0,2]*I3Winc1 + B_2[0,2]*I3J10)*g2_I3 + (B_1[1,2]*I3J10*g2_I3);
        R3 = Rtemp + dt*(A[1,2]*f2_R) + (B_1[0,2]*RWinc1 + B_2[0,2]*RJ10)*g2_R + (B_1[1,2]*RJ10*g2_R);
        D3 = Dtemp + dt*(A[1,2]*f2_D) + (B_1[0,2]*DWinc1 + B_2[0,2]*DJ10)*g2_D + (B_1[1,2]*DJ10*g2_D);
        # V3 = Vtemp + dt*(A[1,2]*f2_V) + (B_1[0,2]*VWinc1 + B_2[0,2]*VJ10)*g2_V + (B_1[1,2]*VJ10*g2_V);
        
        f3_S = fS(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g3_S = gS(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f3_I1 = fI1(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g3_I1 = gI1(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f3_I2 = fI2(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g3_I2 = gI2(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f3_I3 = fI3(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g3_I3 = gI3(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f3_R = fR(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g3_R = gR(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f3_D = fD(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g3_D = gD(S3,I13,I23,I33,R3,D3,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        
        S4 = Stemp + dt*f3_S + (B_1[0,3]*SWinc1 + B_2[0,3]*SJ10)*g3_S + (B_1[1,3]*SWinc1*g3_S + B_1[2,3]*SWinc1*g3_S);
        I14 = I1temp + dt*f3_I1 + (B_1[0,3]*I1Winc1 + B_2[0,3]*I1J10)*g3_I1 + (B_1[1,3]*I1Winc1*g3_I1 + B_1[2,3]*I1Winc1*g3_I1);
        I24 = I2temp + dt*f3_I2 + (B_1[0,3]*I2Winc1 + B_2[0,3]*I2J10)*g3_I2 + (B_1[1,3]*I2Winc1*g3_I2 + B_1[2,3]*I2Winc1*g3_I2);
        I34 = I3temp + dt*f3_I3 + (B_1[0,3]*I3Winc1 + B_2[0,3]*I3J10)*g3_I3 + (B_1[1,3]*I3Winc1*g3_I3 + B_1[2,3]*I3Winc1*g3_I3);
        R4 = Rtemp + dt*f3_R + (B_1[0,3]*RWinc1 + B_2[0,3]*RJ10)*g3_R + (B_1[1,3]*RWinc1*g3_R + B_1[2,3]*RWinc1*g3_R);
        D4 = Dtemp + dt*f3_D + (B_1[0,3]*DWinc1 + B_2[0,3]*DJ10)*g3_D + (B_1[1,3]*DWinc1*g3_D + B_1[2,3]*DWinc1*g3_D);
        # V4 = Vtemp + dt*f3_V + (B_1[0,3]*VWinc1 + B_2[0,3]*VJ10)*g3_V + (B_1[1,3]*VWinc1*g3_V + B_1[2,3]*VWinc1*g3_V);
        
        f4_S = fS(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g4_S = gS(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f4_I1 = fI1(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g4_I1 = gI1(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f4_I2 = fI2(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g4_I2 = gI2(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f4_I3 = fI3(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g4_I3 = gI3(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f4_R = fR(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g4_R = gR(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        f4_D = fD(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        g4_D = gD(S4,I14,I24,I34,R4,D4,Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu);
        
        Stemp = Stemp + dt*((1/6)*f1_S + (1/3)*f2_S + (1/3)*f3_S + (1/6)*f4_S) + (G_1[0]*SWinc1 + G_2[0]*SJ10)*g1_S + (G_1[1]*SWinc1 + G_2[1]*SJ10)*g2_S + (G_1[2]*SWinc1 + G_2[2]*SJ10)*g3_S + (G_1[3]*SWinc1 + G_2[3]*SJ10)*g4_S;
        I1temp = I1temp + dt*((1/6)*f1_I1 + (1/3)*f2_I1 + (1/3)*f3_I1 + (1/6)*f4_I1) + (G_1[0]*I1Winc1 + G_2[0]*I1J10)*g1_I1 + (G_1[1]*I1Winc1 + G_2[1]*I1J10)*g2_I1 + (G_1[2]*I1Winc1 + G_2[2]*I1J10)*g3_I1 + (G_1[3]*I1Winc1 + G_2[3]*I1J10)*g4_I1;
        I2temp = I2temp + dt*((1/6)*f1_I2 + (1/3)*f2_I2 + (1/3)*f3_I2 + (1/6)*f4_I2) + (G_1[0]*I2Winc1 + G_2[0]*I2J10)*g1_I2 + (G_1[1]*I2Winc1 + G_2[1]*I2J10)*g2_I2 + (G_1[2]*I2Winc1 + G_2[2]*I2J10)*g3_I2 + (G_1[3]*I2Winc1 + G_2[3]*I2J10)*g4_I2;
        I3temp = I3temp + dt*((1/6)*f1_I3 + (1/3)*f2_I3 + (1/3)*f3_I3 + (1/6)*f4_I3) + (G_1[0]*I3Winc1 + G_2[0]*I3J10)*g1_I3 + (G_1[1]*I3Winc1 + G_2[1]*I3J10)*g2_I3 + (G_1[2]*I3Winc1 + G_2[2]*I3J10)*g3_I3 + (G_1[3]*I3Winc1 + G_2[3]*I3J10)*g4_I3;
        Rtemp = Rtemp + dt*((1/6)*f1_R + (1/3)*f2_R + (1/3)*f3_R + (1/6)*f4_R) + (G_1[0]*RWinc1 + G_2[0]*RJ10)*g1_R + (G_1[1]*RWinc1 + G_2[1]*RJ10)*g2_R + (G_1[2]*RWinc1 + G_2[2]*RJ10)*g3_R + (G_1[3]*RWinc1 + G_2[3]*RJ10)*g4_R;
        Dtemp = Dtemp + dt*((1/6)*f1_D + (1/3)*f2_D + (1/3)*f3_D + (1/6)*f4_D) + (G_1[0]*DWinc1 + G_2[0]*DJ10)*g1_D + (G_1[1]*DWinc1 + G_2[1]*DJ10)*g2_D + (G_1[2]*DWinc1 + G_2[2]*DJ10)*g3_D + (G_1[3]*DWinc1 + G_2[3]*DJ10)*g4_D;
        
        Ssrk4_2[int(j) - t0] = Stemp;
        I1srk4_2[int(j) - t0] = I1temp;
        I2srk4_2[int(j) - t0] = I2temp;
        I3srk4_2[int(j) - t0] = I3temp;
        Rsrk4_2[int(j) - t0] = Rtemp;
        Dsrk4_2[int(j) - t0] = Dtemp;
        
    return Ssrk4_2, I1srk4_2, I2srk4_2, I3srk4_2, Rsrk4_2, Dsrk4_2

def generate_plot(radio_option, toggle_button_index):
     
    fig_SIRD_projection = go.Figure()
    if model_option == "SIR":
        model_used = "SIR"
    elif model_option == "SIRD":
        model_used = "l-SIRD" if toggle_button7 else "SIRD"
    else:
        model_used = "l-SI3RD" if toggle_button7 else "SI3RD"
    pastplot = data_SI3RD[model_used][radio_option]['pastplot']
    data_S  = data_SI3RD[model_used][radio_option]['fulldata']['susceptible']
    data_I1 = data_SI3RD[model_used][radio_option]['fulldata']['cases_active']
    data_I2 = data_SI3RD[model_used][radio_option]['fulldata']['icu_covid']
    data_I3 = data_SI3RD[model_used][radio_option]['fulldata']['vent_covid']
    data_R  = data_SI3RD[model_used][radio_option]['fulldata']['cases_recovered']
    data_D  = data_SI3RD[model_used][radio_option]['fulldata']['deaths_total'] 
    sim_S   = data_SI3RD[model_used][radio_option]['percentile_S']
    sim_I1  = data_SI3RD[model_used][radio_option]['percentile_I1']
    sim_I2  = data_SI3RD[model_used][radio_option]['percentile_I2']
    sim_I3  = data_SI3RD[model_used][radio_option]['percentile_I3']
    sim_R   = data_SI3RD[model_used][radio_option]['percentile_R']
    sim_D   = data_SI3RD[model_used][radio_option]['percentile_D']
    
    # Apply toggles
    if toggle_button_index[0]: #Susceptible
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_S[:,0] ,name = "Q1 Susceptible", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Susceptible'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_S[:,2] ,name = "Q3 Susceptible", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(47,79,79,0.5)',legendgroup = 'Susceptible'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_S[:,1] , name = "Susceptible", marker = dict(color = 'darkslategray'),legendgroup = 'Susceptible'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = data_S, mode='lines',name = "Susceptible data", line = dict(color = 'darkkhaki', width = 3, dash = 'dash'),legendgroup = 'Susceptible'))

    if toggle_button_index[1]: #I1
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I1[:,0] ,name = "Q1 Infected 1", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I1[:,2] ,name = "Q3 Infected 1", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(90, 130, 209,0.5)',legendgroup = 'Infected'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I1[:,1] ,name = "Infected 1",marker = dict(color = 'rgba(14, 60, 150, 1)'),legendgroup = 'Infected'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = data_I1, mode='lines',name = "Infected data 1", line = dict(color = 'rgba(48, 63, 122, 1)', width =3, dash ='dash'),legendgroup = 'Infected'))        
    
    if toggle_button_index[2] and model_option == "SI₃RD": #I2
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I2[:,0] ,name = "Q1 Infected 2", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected 2'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I2[:,2] ,name = "Q3 Infected 2", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(157, 90, 209, 0.5)',legendgroup = 'Infected 2'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I2[:,1] ,name = "Infected 2",marker = dict(color = 'mediumslateblue'),legendgroup = 'Infected 2'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = data_I2, mode='lines',name = "Infected data 2", line = dict(color = 'darkorchid', width =3, dash ='dash'),legendgroup = 'Infected 2'))

    if toggle_button_index[3] and model_option == "SI₃RD": #I3
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I3[:,0] ,name = "Q1 Infected 3", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected 3'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I3[:,2] ,name = "Q3 Infected 3", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(242, 166, 111, 0.5)',legendgroup = 'Infected 3'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_I3[:,1] ,name = "Infected 3",marker = dict(color = 'rgba(255, 189, 46, 1)'),legendgroup = 'Infected 3'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = data_I3, mode='lines',name = "Infected data 3", line = dict(color = 'rgba(194, 68, 0, 1)', width =3, dash ='dash'),legendgroup = 'Infected 3'))

    if toggle_button_index[4]: #Recovered
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_R[:,0] ,name = "Q1 Recovered", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Recovered'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_R[:,2] ,name = "Q3 Recovered", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(124,252,0,0.5)',legendgroup = 'Recovered'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_R[:,1] ,name = "Recovered",marker = dict(color = 'lawngreen'),legendgroup = 'Recovered'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = data_R, mode='lines', name = "Recovered data", line = dict(color = 'olive', width = 3, dash='dash'),legendgroup = 'Recovered'))

    if toggle_button_index[5] and (model_option == "SIRD" or model_option == "SI₃RD"): #Death
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_D[:,0] ,name = "Q1 Death", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Death'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_D[:,2] ,name = "Q3 Death", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(128, 0, 0,0.5)',legendgroup = 'Death'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = sim_D[:,1] ,name = "Death",marker = dict(color = 'maroon'),legendgroup = 'Death'))
        fig_SIRD_projection.add_trace(go.Scatter(x = pastplot, y = data_D, mode='lines',name = "Death data", line = dict(color = 'tomato', width =3, dash = 'dash'),legendgroup = 'Death'))

    fig_SIRD_projection.update_layout(
        xaxis_title='t(days)',
        yaxis_title='Population',
        height=620,
        width=1500,
        margin=dict(l=20, r=20, t=20, b=20),
        legend = dict(x=0.01, y = 0.95),
        legend_traceorder="normal"
    )

    return fig_SIRD_projection

def date_to_integer(input_date, first_date):
    # Convert input_date to datetime object
    if isinstance(input_date, datetime):
        input_datetime = input_date
    elif isinstance(input_date, date):
        input_datetime = datetime.combine(input_date, datetime.min.time())
    else:
        raise ValueError("Unsupported input_date type")

    # Convert first_date to datetime object
    if isinstance(first_date, datetime):
        first_datetime = first_date
    elif isinstance(first_date, date):
        first_datetime = datetime.combine(first_date, datetime.min.time())
    else:
        raise ValueError("Unsupported first_date type")

    # Calculate the difference in days
    dateint = input_datetime - first_datetime

    return dateint.days
               
def parameter_func(j,radio_option, date_inputs,strictness_inputs):
    if toggle_button7 == "True":
        mu = 0.000021;
    else:
        mu = 0;
    
    if model_option == "SIR":
        model_used = "SIR"
        mu = 0;
    elif model_option == "SIRD":
        model_used = "l-SIRD" if toggle_button7 else "SIRD"
    else:
        model_used = "l-SI3RD" if toggle_button7 else "SI3RD"

    sdate2 = data_SI3RD[model_used][radio_option]['sdate2']
    futplot = data_SI3RD[model_used][radio_option]['futplot']
    pars_r1 = data_SI3RD[model_used][radio_option]['pars_r1']
    pars_r2 = data_SI3RD[model_used][radio_option]['pars_r2']
    pars_r3 = data_SI3RD[model_used][radio_option]['pars_r3']
    pars_d1 = data_SI3RD[model_used][radio_option]['pars_d1']
    pars_d2 = data_SI3RD[model_used][radio_option]['pars_d2']
    pars_d3 = data_SI3RD[model_used][radio_option]['pars_d3']
    pars_p1 = data_SI3RD[model_used][radio_option]['pars_p1']
    pars_p2 = data_SI3RD[model_used][radio_option]['pars_p2']
    pars_p3 = data_SI3RD[model_used][radio_option]['pars_p3']
        
    Beta1,Beta2,Beta3 = 0, 0, 0;
    r1,r2,r3 = 0, 0, 0;
    d1,d2,d3 = 0, 0, 0;
    gamma,sigma = 0, 0.05;
    p1, p2, p3 = 1, 0, 0;
    
    npidate_1 = date_to_integer(date_inputs[0], sdate2)
    npidate_2 = date_to_integer(date_inputs[1], sdate2)
    npidate_3 = date_to_integer(date_inputs[2], sdate2)
    npidate_4 = date_to_integer(date_inputs[3], sdate2)
    npidate_5 = date_to_integer(date_inputs[4], sdate2)
    npidate_6 = date_to_integer(date_inputs[5], sdate2)
    npidate_7 = date_to_integer(date_inputs[6], sdate2)
    npidate_8 = date_to_integer(date_inputs[7], sdate2)
    
    futbetavalues = {
        'Malaysia': {'Strict':[0.039616165,0.027115,0.029983], 'Moderate': [0.070182669,0.032846,0.0363214], 'Loose':[0.0976,0.06495,0.1052]},
        'Canada': {'Strict': [0.115,0.00085,0.000275], 'Moderate': [0.175,0.0015,0.000425], 'Loose': [0.19,0.0024,0.000750]},
        'Italy': {'Strict': [0.025,0.001,0.0001], 'Moderate': [0.075,0.045,0.00001], 'Loose': [0.09,0.0001,0.00475]},
        'Slovakia': {'Strict': [0.008,0.01,0.00025], 'Moderate': [0.05,0.0025,0.029983], 'Loose': [0.082,0.015,0.000001]},
        'SouthAfrica': {'Strict': [0.045,0.0007,0.0003], 'Moderate': [0.1,0.00055,0.029983], 'Loose': [0.13,0.005,0.0009]},
    }
    T_fut = len(futplot)
    
    if j <= npidate_1:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[0]]
    elif j > npidate_1 and j <= npidate_2:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[1]]
    elif j > npidate_2 and j <= npidate_3:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[2]]
    elif j > npidate_3 and j <= npidate_4:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[3]]
    elif j > npidate_4 and j <= npidate_5:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[4]]
    elif j > npidate_5 and j <= npidate_6:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[5]]
    elif j > npidate_6 and j <= npidate_7:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[6]]
    elif j > npidate_7 and j <= npidate_8:
            Beta1, Beta2, Beta3 = futbetavalues[radio_option][strictness_inputs[7]]
    
    T_fut = len(futplot)
    r1 = log_func(j + T_fut,*pars_r1)
    if model_option == "SIRD" or model_option == "SI₃RD":
        d1 = log_func(j + T_fut,*pars_d1)
        gamma = data_SI3RD[model_used][radio_option]['mcmcresult']['gamma'].mean()
    if model_option == "SI₃RD":
        r2 = log_func(j + T_fut,*pars_r2)
        r3 = log_func(j + T_fut,*pars_r3)
        d2 = log_func(j + T_fut,*pars_d2)
        d3 = log_func(j + T_fut,*pars_d3)
        p1 = log_func(j + T_fut,*pars_p1)
        p2 = log_func(j + T_fut,*pars_p2)
        p3 = log_func(j + T_fut,*pars_p3)
        
    return Beta1,Beta2,Beta3,r1,r2,r3,d1,d2,d3,gamma,sigma,p1,p2,p3,mu

def generate_hist(radio_option):
    fig_beta1_dist = go.Figure()
    fig_beta2_dist = go.Figure()
    fig_beta3_dist = go.Figure()
    
    if radio_option == "Malaysia":
        dt_mcmcresult = data_SI3RD["SI3RD"][radio_option]['mcmcresult'] 
        custom_colors_1 = ['#6ef74f']*7 + ['#4f95f7']*8 + ['#942c8b']*7
        custom_colors_2 = ['#6ef74f']*5 + ['#4f95f7']*6 + ['#942c8b']*12
        custom_colors_3 = ['#6ef74f']*5 + ['#4f95f7']*3 + ['#942c8b']*4
        xlim_1 = [0,0.15]; xlim_2 = [0,0.0025]; xlim_3 = [0,0.0018];
    elif radio_option == "Canada":
        dt_mcmcresult = data_SI3RD["SI3RD"][radio_option]['mcmcresult'] 
        custom_colors_1 = ['#6ef74f']*6 + ['#4f95f7']*3 + ['#942c8b']*2
        custom_colors_2 = ['#6ef74f']*6 + ['#4f95f7']*4 + ['#942c8b']*4
        custom_colors_3 = ['#6ef74f']*6 + ['#4f95f7']*4 + ['#942c8b']*7
        xlim_1 = [0.1,0.21]; xlim_2 = [0,0.0038]; xlim_3 = [0,0.00085];
    elif radio_option == "Italy":
        dt_mcmcresult = data_SI3RD["SI3RD"][radio_option]['mcmcresult'] 
        custom_colors_1 = ['#6ef74f']*6 + ['#4f95f7']*3 + ['#942c8b']*2
        custom_colors_2 = ['#6ef74f']*6 + ['#4f95f7']*4 + ['#942c8b']*4
        custom_colors_3 = ['#6ef74f']*6 + ['#4f95f7']*4 + ['#942c8b']*7
        xlim_1 = [0,0.1]; xlim_2 = [0,0.0475]; xlim_3 = [0,0.0053];
    elif radio_option == "Slovakia":
        dt_mcmcresult = data_SI3RD["SI3RD"][radio_option]['mcmcresult'] 
        custom_colors_1 = ['#6ef74f']*4 + ['#4f95f7']*8 + ['#942c8b']*4
        custom_colors_2 = ['#6ef74f']*3 + ['#4f95f7']*2 + ['#942c8b']*14
        custom_colors_3 = ['#6ef74f']*3 + ['#4f95f7']*4 + ['#942c8b']*3
        xlim_1 = [0,0.085]; xlim_2 = [0,0.019]; xlim_3 = [0,0.00285];
    elif radio_option == "South Africa":
        dt_mcmcresult = data_SI3RD["SI3RD"][radio_option]['mcmcresult'] 
        custom_colors_1 = ['#6ef74f']*5 + ['#4f95f7']*4 + ['#942c8b']*4
        custom_colors_2 = ['#6ef74f']*3 + ['#4f95f7']*6 + ['#942c8b']*3
        custom_colors_3 = ['#6ef74f']*4 + ['#4f95f7']*4 + ['#942c8b']*3
        xlim_1 = [0.03,0.14]; xlim_2 = [0,0.006]; xlim_3 = [0,0.0011];
    else:
        dt_mcmcresult = data_SI3RD["SI3RD"][radio_option]['mcmcresult'] 
        custom_colors_1 = ['#6ef74f']*7 + ['#4f95f7']*8 + ['#942c8b']*7
        custom_colors_2 = ['#6ef74f']*5 + ['#4f95f7']*6 + ['#942c8b']*12
        custom_colors_3 = ['#6ef74f']*5 + ['#4f95f7']*3 + ['#942c8b']*4
        xlim_1 = [0,0.15]; xlim_2 = [0,0.0025]; xlim_3 = [0,0.0018];
    
    fig_beta1_dist.add_trace(go.Histogram(x=dt_mcmcresult['beta1'], marker = dict(color = custom_colors_1),opacity=0.75))
    fig_beta1_dist.update_layout(height=300, width = 500)
    fig_beta1_dist.update_layout(xaxis_title="Infectious rate, beta 1",yaxis_title="Frequency",)
    fig_beta1_dist.update_xaxes(range=xlim_1)
    if model_option == "SI₃RD":
        fig_beta2_dist.add_trace(go.Histogram(x=dt_mcmcresult['beta2'], marker = dict(color = custom_colors_2),opacity=0.75))
        fig_beta2_dist.update_layout(height=300, width = 500)
        fig_beta2_dist.update_layout(xaxis_title="Infectious rate, beta 2",yaxis_title="Frequency",)
        fig_beta2_dist.update_xaxes(range=xlim_2)
        
        fig_beta3_dist.add_trace(go.Histogram(x=dt_mcmcresult['beta3'], marker = dict(color = custom_colors_3),opacity=0.75))
        fig_beta3_dist.update_layout(height=300, width = 500)
        fig_beta3_dist.update_layout(xaxis_title="Infectious rate, beta 3",yaxis_title="Frequency",)
        fig_beta3_dist.update_xaxes(range=xlim_3)
    
    fig_beta1_dist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    if model_option == "SI₃RD":
        fig_beta2_dist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        fig_beta3_dist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    
    return fig_beta1_dist, fig_beta2_dist, fig_beta3_dist

def generate_predict_plot(radio_option, toggle_button_index,date_inputs,strictness_inputs):
    
    fig_SIRD_prediction = go.Figure()
    
    if model_option == "SIR":
        model_used = "SIR"
    elif model_option == "SIRD":
        model_used = "l-SIRD" if toggle_button7 else "SIRD"
    else:
        model_used = "l-SI3RD" if toggle_button7 else "SI3RD"
    
    futplot = data_SI3RD[model_used][radio_option]['futplot']
    Szero = data_SI3RD[model_used][radio_option]['fulldata']['susceptible'][-1]
    I1zero = data_SI3RD[model_used][radio_option]['fulldata']['cases_active'][-1]
    Rzero = data_SI3RD[model_used][radio_option]['fulldata']['cases_recovered'][-1]
    if model_option == "SI₃RD":
        I2zero = data_SI3RD[model_used][radio_option]['fulldata']['icu_covid'][-1]
        I3zero = data_SI3RD[model_used][radio_option]['fulldata']['vent_covid'][-1]
    else:
        I2zero = 0
        I3zero = 0
    if model_option == "SI₃RD" or model_option == "SIRD":
        Dzero = data_SI3RD[model_used][radio_option]['fulldata']['deaths_total'][-1]
    else:
        Dzero = 0
    
    Ptotal = Szero+I1zero+I2zero+I3zero+Rzero+Dzero
    Ssrk4_2, I1srk4_2,I2srk4_2,I3srk4_2, Rsrk4_2, Dsrk4_2 = SI3RD_func(0,1,len(futplot),500,Szero,I1zero,I2zero,I3zero,Rzero,Dzero,Ptotal,radio_option, toggle_button_index,date_inputs,strictness_inputs)
    
    percentileSfut = np.ones((len(futplot),3));
    percentileI1fut = np.ones((len(futplot),3));
    percentileI2fut = np.ones((len(futplot),3));
    percentileI3fut = np.ones((len(futplot),3));
    percentileRfut = np.ones((len(futplot),3));
    percentileDfut = np.ones((len(futplot),3));

    for j in np.linspace(0,len(futplot)-1,len(futplot)):
        percentileSfut[j.astype(int)] = (pd.Series(Ssrk4_2[j.astype(int),:])).quantile([lowper,mean,highper]);
        percentileI1fut[j.astype(int)] = (pd.Series(I1srk4_2[j.astype(int),:])).quantile([lowper,mean,highper]);
        percentileI2fut[j.astype(int)] = (pd.Series(I2srk4_2[j.astype(int),:])).quantile([lowper,mean,highper]);
        percentileI3fut[j.astype(int)] = (pd.Series(I3srk4_2[j.astype(int),:])).quantile([lowper,mean,highper]);
        percentileRfut[j.astype(int)] = (pd.Series(Rsrk4_2[j.astype(int),:])).quantile([lowper,mean,highper]);
        percentileDfut[j.astype(int)] = (pd.Series(Dsrk4_2[j.astype(int),:])).quantile([lowper,mean,highper]);
    
    xlim = max(
    date_to_integer(date_inputs[0], futplot[0]),
    date_to_integer(date_inputs[1], futplot[0]),
    date_to_integer(date_inputs[2], futplot[0]),
    date_to_integer(date_inputs[3], futplot[0]),
    date_to_integer(date_inputs[4], futplot[0]),
    date_to_integer(date_inputs[5], futplot[0]),
    date_to_integer(date_inputs[6], futplot[0]),
    date_to_integer(date_inputs[7], futplot[0])
    )
    
    if toggle_button_index[0]: #Susceptible
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileSfut[:xlim,0] ,name = "Q1 Susceptible", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Susceptible'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileSfut[:xlim,2] ,name = "Q3 Susceptible", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(47,79,79,0.5)',legendgroup = 'Susceptible'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileSfut[:xlim,1] , name = "Susceptible", marker = dict(color = 'darkslategray'),legendgroup = 'Susceptible'))
        
    if toggle_button_index[1]: #I1
        #fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='y2 (cosine)'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI1fut[:xlim,0] ,name = "Q1 Infected 1", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI1fut[:xlim,2] ,name = "Q3 Infected 1", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(90, 130, 209,0.5)',legendgroup = 'Infected'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI1fut[:xlim,1] ,name = "Infected 1",marker = dict(color = 'rgba(14, 60, 150, 1)'),legendgroup = 'Infected'))
        
    if toggle_button_index[2] and model_option == "SI₃RD": #I2
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI2fut[:xlim,0] ,name = "Q1 Infected 2", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected 2'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI2fut[:xlim,2] ,name = "Q3 Infected 2", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(157, 90, 209, 0.5)',legendgroup = 'Infected 2'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI2fut[:xlim,1] ,name = "Infected 2",marker = dict(color = 'mediumslateblue'),legendgroup = 'Infected 2'))
        
    if toggle_button_index[3] and model_option == "SI₃RD": #I3
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI3fut[:xlim,0] ,name = "Q1 Infected 3", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected 3'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI3fut[:xlim,2] ,name = "Q3 Infected 3", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(242, 166, 111, 0.5)',legendgroup = 'Infected 3'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileI3fut[:xlim,1] ,name = "Infected 3",marker = dict(color = 'rgba(255, 189, 46, 1)'),legendgroup = 'Infected 3'))
        
    if toggle_button_index[4]: #Recovered
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileRfut[:xlim,0] ,name = "Q1 Recovered", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Recovered'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileRfut[:xlim,2] ,name = "Q3 Recovered", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(124,252,0,0.5)',legendgroup = 'Recovered'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileRfut[:xlim,1] ,name = "Recovered",marker = dict(color = 'lawngreen'),legendgroup = 'Recovered'))
        
    if toggle_button_index[5] and (model_option == "SIRD" or model_option == "SI₃RD"): #Death
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileDfut[:xlim,0] ,name = "Q1 Death", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Death'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileDfut[:xlim,2] ,name = "Q3 Death", showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(128, 0, 0,0.5)',legendgroup = 'Death'))
        fig_SIRD_prediction.add_trace(go.Scatter(x = futplot[:xlim], y = percentileDfut[:xlim,1] ,name = "Death",marker = dict(color = 'maroon'),legendgroup = 'Death'))
    
    fig_SIRD_prediction.update_layout(
        xaxis_title='t(days)',
        yaxis_title='Population',
        height=600,
        width=1350,
        margin=dict(l=20, r=20, t=20, b=20),
        legend = dict(x=0.01, y = 0.95),
        legend_traceorder="normal"
    )
    
    return fig_SIRD_prediction
def generate_trend(radio_option):
    
    fig_r1_trend = go.Figure()
    fig_r2_trend = go.Figure()
    fig_r3_trend = go.Figure()
    fig_d1_trend = go.Figure()
    fig_d2_trend = go.Figure()
    fig_d3_trend = go.Figure()
    fig_p1_trend = go.Figure()
    fig_p2_trend = go.Figure()
    fig_p3_trend = go.Figure()
    
    if model_option == "SIR":
        model_used = "SIR"
    elif model_option == "SIRD":
        model_used = "l-SIRD" if toggle_button7 else "SIRD"
    else:
        model_used = "l-SI3RD" if toggle_button7 else "SI3RD"

    dt_mcmcresult = data_SI3RD[model_used][radio_option]['mcmcresult'][1:]
    pastplot = data_SI3RD[model_used][radio_option]['pastplot'][1:]
    pars_r1 = data_SI3RD[model_used][radio_option]['pars_r1']
    pars_r2 = data_SI3RD[model_used][radio_option]['pars_r2']
    pars_r3 = data_SI3RD[model_used][radio_option]['pars_r3']
    pars_d1 = data_SI3RD[model_used][radio_option]['pars_d1']
    pars_d2 = data_SI3RD[model_used][radio_option]['pars_d2']
    pars_d3 = data_SI3RD[model_used][radio_option]['pars_d3']
    pars_p1 = data_SI3RD[model_used][radio_option]['pars_p1']
    pars_p2 = data_SI3RD[model_used][radio_option]['pars_p2']
    pars_p3 = data_SI3RD[model_used][radio_option]['pars_p3']
            
    fig_r1_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['r1'] ,name = "Recovery rate 1",marker = dict(color = 'rgba(124, 209, 13, 1)')))
    fig_r1_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_r1),showlegend= False,mode = 'lines', line = dict(color = 'rgba(71, 117, 11, 1)', width = 3, dash='dash')))
    
    if model_option == "SIRD" or model_option == "SI₃RD":
        fig_d1_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['d1'] ,name = "Fatality rate 1",marker = dict(color = 'rgba(169, 16, 230, 1)')))
        fig_d1_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_d1),showlegend= False,mode = 'lines', line = dict(color = 'rgba(76, 12, 135, 1)', width = 3, dash='dash')))
    
    if model_option == "SI₃RD":
        fig_r2_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['r2'] ,name = "Recovery rate 2",marker = dict(color = 'rgba(48, 232, 23, 1)')))
        fig_r2_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_r2),showlegend= False,mode = 'lines', line = dict(color = 'rgba(33, 166, 15, 1)', width = 3, dash='dash')))
        fig_r3_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['r3'] ,name = "Recovery rate 3",marker = dict(color = 'rgba(14, 227, 145, 1)')))
        fig_r3_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_r3),showlegend= False,mode = 'lines', line = dict(color = 'rgba(9, 179, 113, 1)', width = 3, dash='dash')))
    
        fig_d2_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['d2'] ,name = "Fatality rate 2",marker = dict(color = 'rgba(227, 16, 41, 1)')))
        fig_d2_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_d2),showlegend= False,mode = 'lines', line = dict(color = 'rgba(158, 11, 28, 1)', width = 3, dash='dash')))
        fig_d3_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['d3'] ,name = "Fatality rate 3",marker = dict(color = 'rgba(242, 77, 12, 1)')))
        fig_d3_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_d3),showlegend= False,mode = 'lines', line = dict(color = 'rgba(166, 53, 8, 1)', width = 3, dash='dash')))
        
        fig_p1_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['p1'] ,name = "Proportion rate 1",marker = dict(color = 'rgba(66, 115, 125, 1)')))
        fig_p1_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_p1),showlegend= False,mode = 'lines', line = dict(color = 'rgba(38, 66, 71, 1)', width = 3, dash='dash')))
        fig_p2_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['p2'] ,name = "Proportion rate 2",marker = dict(color = 'rgba(159, 179, 34, 1)')))
        fig_p2_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_p2),showlegend= False,mode = 'lines', line = dict(color = 'rgba(100, 112, 20, 1)', width = 3, dash='dash')))
        fig_p3_trend.add_trace(go.Scatter(x = pastplot, y = dt_mcmcresult['p3'] ,name = "Proportion rate 3",marker = dict(color = 'rgba(133, 133, 133, 1)')))
        fig_p3_trend.add_trace(go.Scatter(x = pastplot, y = log_func(np.linspace(1,len(pastplot),len(pastplot)+1),*pars_p3),showlegend= False,mode = 'lines', line = dict(color = 'rgba(87, 87, 87, 1)', width = 3, dash='dash')))
    
    fig_r1_trend.update_layout(yaxis_title="Recovery rate, r 1", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_d1_trend.update_layout(yaxis_title="Fatality rate, delta 1", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_r2_trend.update_layout(yaxis_title="Recovery rate, r 2", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_r3_trend.update_layout(yaxis_title="Recovery rate, r 3", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_d2_trend.update_layout(yaxis_title="Fatality rate, delta 2", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_d3_trend.update_layout(yaxis_title="Fatality rate, delta 3", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_p1_trend.update_layout(yaxis_title="Proportion rate, p 1", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_p2_trend.update_layout(yaxis_title="Proportion rate, p 2", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    fig_p3_trend.update_layout(yaxis_title="Proportion rate, p 3", xaxis_title="t(days)",height=300, width = 600,legend=dict(x=0.8,y=0.95))
    
    fig_r1_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_d1_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_r2_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_r3_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_d2_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_d3_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_p1_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_p2_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig_p3_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    
    return fig_r1_trend, fig_r2_trend, fig_r3_trend, fig_d1_trend, fig_d2_trend, fig_d3_trend, fig_p1_trend, fig_p2_trend, fig_p3_trend

st.title("Pandemic Simulation SI₃RD")

with st.container():
    # Define column widths as a percentage of the page width
    col1, col2 = st.columns([1.5, 10], gap='small')

    # Left column (smaller) with radio buttons and toggle buttons
    with col1:
        #st.write("#")
        model_option = st.radio("Model:", ["SIR", "SIRD", "SI₃RD"], index = 2)
        toggle_button7 = st.toggle("Natural birth and Fatality", value=False)
        
        radio_option = st.radio("Country:", ["Malaysia", "Canada", "Italy", "Slovakia", "SouthAfrica"])
        st.caption('Compartment:')
        if model_option == "SIR":
            toggle_button1 = st.checkbox("Susceptible", value=False)
            toggle_button2 = st.checkbox("Infected 1", value=True)
            toggle_button3 = st.checkbox("Infected 2", value=False)
            toggle_button4 = st.checkbox("Infected 3", value=False)
            toggle_button5 = st.checkbox("Recovered", value=True)
            toggle_button6 = st.checkbox("Death", value=False)
        elif model_option == "SIRD":
            toggle_button1 = st.checkbox("Susceptible", value=False)
            toggle_button2 = st.checkbox("Infected 1", value=True)
            toggle_button3 = st.checkbox("Infected 2", value=False)
            toggle_button4 = st.checkbox("Infected 3", value=False)
            toggle_button5 = st.checkbox("Recovered", value=True)
            toggle_button6 = st.checkbox("Death", value=True)
        else:
            toggle_button1 = st.checkbox("Susceptible", value=False)
            toggle_button2 = st.checkbox("Infected 1", value=True)
            toggle_button3 = st.checkbox("Infected 2", value=True)
            toggle_button4 = st.checkbox("Infected 3", value=True)
            toggle_button5 = st.checkbox("Recovered", value=True)
            toggle_button6 = st.checkbox("Death", value=True)
         
        toggle_button_index = [toggle_button1,toggle_button2,toggle_button3,toggle_button4,toggle_button5,toggle_button6]
    with col2:
        # Generate and display the plot
        fig_SIRD_projection = generate_plot(radio_option, toggle_button_index)
        st.plotly_chart(fig_SIRD_projection)

# Additional content outside containers
with st.container():
    # Define column widths as a percentage of the page width
    col1, col2, col3 = st.columns([0.8,1,1], gap='small')
    
    with col1:
        fig_beta1_dist, fig_beta2_dist, fig_beta3_dist = generate_hist(radio_option)
        tab1, tab2, tab3 = st.tabs(["Infectious rate 1", "Infectious rate 2", "Infectious rate 3"])
        with tab1:
            st.plotly_chart(fig_beta1_dist)
        with tab2:
            st.plotly_chart(fig_beta2_dist)
        with tab3:
            st.plotly_chart(fig_beta3_dist)
        
    fig_r1_trend, fig_r2_trend, fig_r3_trend, fig_d1_trend, fig_d2_trend, fig_d3_trend, fig_p1_trend, fig_p2_trend, fig_p3_trend = generate_trend(radio_option)
        
    with col2:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Recovery rate 1", "Recovery rate 2", "Recovery rate 3", "Fatality rate 1", "Fatality rate 2", "Fatality rate 3"])
        with tab1:
            st.plotly_chart(fig_r1_trend)
        with tab2:
            st.plotly_chart(fig_r2_trend)
        with tab3:
            st.plotly_chart(fig_r3_trend)
        with tab4:
            st.plotly_chart(fig_d1_trend)
        with tab5:
            st.plotly_chart(fig_d2_trend)
        with tab6:
            st.plotly_chart(fig_d3_trend)
                    
    with col3:
        tab1, tab2, tab3 = st.tabs(["Proportion rate 1", "Proportion rate 2", "Proportion rate 3"])
        with tab1:
            st.plotly_chart(fig_p1_trend)
        with tab2:
            st.plotly_chart(fig_p2_trend)
        with tab3:
            st.plotly_chart(fig_p3_trend)

with st.container():
    # Define column widths as a percentage of the page width
    col1, col2 = st.columns([1.4, 5], gap='small')
    futplot = data_SI3RD["SI3RD"][radio_option]['futplot']
    
    with col1:
        st.title("Prediction Tool")
        st.write("Input date and NPI strictness:")
        date_inputs = []
        strictness_inputs = []
        #strictness_values = []
        for i in range(8):
            with st.container():
                col21, col22 = st.columns([1, 1], gap='small')
                with col21:
                    date_input = st.date_input(f"Date {i+1}:", value=futplot[1])
                    date_inputs.append(date_input)
                with col22:
                    strictness_input  = st.selectbox(f"NPI class {i + 1}:", ["Strict", "Moderate", "Loose"])
                    strictness_inputs.append(strictness_input)
                #with col3:
                    #strictness_value  = st.text_input(f"NPI val {i + 1}: (0-1)")
                    #strictness_values.append(strictness_value)
                    #strictness_values = 0;

    with col2:
        st.write(":")
        fig_SIRD_prediction = generate_predict_plot(radio_option, toggle_button_index,date_inputs,strictness_inputs)
        st.plotly_chart(fig_SIRD_prediction)