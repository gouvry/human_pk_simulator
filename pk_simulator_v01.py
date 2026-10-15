import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title='Human PK Simulator', layout='wide')
st.title('Human PK Simulator')
st.caption('One-compartment oral PK with first-order absorption and elimination.')

with st.sidebar:
    st.header('Human PK inputs')
    cl_ml_min_kg = st.number_input('Plasma clearance (mL/min/kg)', min_value=0.001, value=5.0, step=0.1)
    vd_l_kg = st.number_input('Volume of distribution (L/kg)', min_value=0.001, value=1.0, step=0.1)
    f_pct = st.number_input('Bioavailability (%)', min_value=0.1, max_value=100.0, value=70.0, step=1.0)
    ka = st.number_input('ka (h^-1)', min_value=0.001, value=1.0, step=0.1)
    mw = st.number_input('Molecular weight (g/mol)', min_value=1.0, value=450.0, step=1.0)
    fup = st.number_input('Fraction unbound in plasma (fu,p)', min_value=0.000001, max_value=1.0, value=0.10, step=0.01, format='%.4f')
    kpuu = st.number_input('Kp,uu (unbound brain / unbound plasma)', min_value=0.000001, value=1.0, step=0.1, format='%.3f')
    bw = st.number_input('Body weight (kg)', min_value=1.0, value=70.0, step=1.0)
    dose = st.number_input('Dose per administration (mg)', min_value=0.001, value=100.0, step=10.0)
    regimen = st.selectbox('Regimen', ['QD', 'BID'])
    days = st.slider('Simulation days', 1, 28, 7)
    ic50 = st.number_input('IC50 (nM)', min_value=0.001, value=100.0, step=10.0)

cl = cl_ml_min_kg * 60.0 / 1000.0 * bw
vd = vd_l_kg * bw
kel = cl / vd
f = f_pct / 100.0
tau = 24.0 if regimen == 'QD' else 12.0
if abs(ka-kel) < 1e-8:
    st.error('ka is effectively equal to kel; choose a slightly different value.')
    st.stop()

def single_c(t, dose_mg):
    t = np.asarray(t, dtype=float)
    return f*dose_mg*ka/(vd*(ka-kel))*(np.exp(-kel*t)-np.exp(-ka*t))

def profile(dose_mg, total_h, dt=0.02):
    t = np.arange(0.0, total_h+dt/2, dt)
    c = np.zeros_like(t)
    for td in np.arange(0.0, total_h+1e-9, tau):
        mask = t >= td
        c[mask] += single_c(t[mask]-td, dose_mg)
    return t, c

def mgL_to_nm(c): return c*1e6/mw

def window_metrics(t, c, start, end):
    mask = (t >= start) & (t <= end+1e-9)
    tw, cw = t[mask], c[mask]
    auc = np.trapezoid(cw, tw)
    scheduled = np.arange(math.ceil(start/tau)*tau, end+1e-9, tau)
    predose=[]
    for td in scheduled:
        if td <= start+1e-9: continue
        idx=np.searchsorted(t,td)-1
        if idx>=0: predose.append(c[idx])
    return {'Cmax':float(np.max(cw)), 'Ctrough':float(min(predose)) if predose else float(cw[-1]), 'Cavg':auc/(end-start), 'AUC24':auc}

def avg_inhibition(t,c,start,end,factor):
    mask=(t>=start)&(t<=end+1e-9)
    tw,cw=t[mask],c[mask]
    free_nm=mgL_to_nm(cw)*factor
    return float(np.trapezoid(free_nm/(free_nm+ic50),tw)/(end-start))

total_h=max(24.0,days*24.0)
t,c=profile(dose,total_h)
last_start=total_h-24.0
first=window_metrics(t,c,0.0,24.0)
last=window_metrics(t,c,last_start,total_h)

st.subheader('Concentration-time profile')
unit=st.radio('Concentration units',['nM','ng/mL'],horizontal=True)
st.markdown('**Profiles to display**')
show_total=st.checkbox('Total plasma',True)
show_free_plasma=st.checkbox('Free plasma',True)
show_free_brain=st.checkbox('Free brain',True)
base_y=mgL_to_nm(c) if unit=='nM' else c*1000.0
fig=go.Figure()
if show_total: fig.add_trace(go.Scatter(x=t/24,y=base_y,mode='lines',name='Total plasma'))
if show_free_plasma: fig.add_trace(go.Scatter(x=t/24,y=base_y*fup,mode='lines',name='Free plasma'))
if show_free_brain: fig.add_trace(go.Scatter(x=t/24,y=base_y*fup*kpuu,mode='lines',name='Free brain'))
fig.update_layout(xaxis_title='Time (days)',yaxis_title=f'Concentration ({unit})',hovermode='x unified')
st.plotly_chart(fig,use_container_width=True)

def display_metrics(m,label,start,end):
    conv=(lambda x:mgL_to_nm(x)) if unit=='nM' else (lambda x:x*1000.0)
    auc_conv=mgL_to_nm(m['AUC24']) if unit=='nM' else m['AUC24']*1000.0
    st.markdown(f'**{label}**')
    selected=[]
    if show_total:selected.append(('Total plasma',1.0))
    if show_free_plasma:selected.append(('Free plasma',fup))
    if show_free_brain:selected.append(('Free brain',fup*kpuu))
    if not selected: st.info('Select at least one profile to display concentration metrics.')
    for name,factor in selected:
        st.caption(name)
        a,b,c1,d=st.columns(4)
        a.metric('Cmax',f"{conv(m['Cmax']*factor):,.0f} {unit}")
        b.metric('Ctrough',f"{conv(m['Ctrough']*factor):,.0f} {unit}")
        c1.metric('Cavg',f"{conv(m['Cavg']*factor):,.0f} {unit}")
        d.metric('AUC0-24',f"{auc_conv*factor:,.0f} {unit}·h")
    x,y=st.columns(2)
    x.metric('Average inhibition, free plasma',f'{100*avg_inhibition(t,c,start,end,fup):.1f}%')
    y.metric('Average inhibition, free brain',f'{100*avg_inhibition(t,c,start,end,fup*kpuu):.1f}%')

display_metrics(first,'First 24 hours',0.0,24.0)
display_metrics(last,'Final complete 24-hour window',last_start,total_h)

st.divider(); st.subheader('Dose calculator')
target_site=st.radio('Target site',['Plasma','Brain'],horizontal=True)
site_factor=fup if target_site=='Plasma' else fup*kpuu
mode=st.selectbox('Dose target (free concentration)',['Ctrough','Cavg','Cmax','Average inhibition'])
if mode=='Average inhibition':
    target=st.number_input('Target average inhibition (%)',min_value=0.1,max_value=99.9,value=80.0)
else:
    target_unit=st.selectbox('Target units',['nM','ng/mL'])
    target=st.number_input(f'Target free {target_site.lower()} {mode} ({target_unit})',min_value=0.001,value=100.0)

def final_metric_for_dose(d):
    tt,cc=profile(d,total_h)
    mm=window_metrics(tt,cc,last_start,total_h)
    if mode=='Average inhibition': return 100*avg_inhibition(tt,cc,last_start,total_h,site_factor)
    val=mm[mode]*site_factor
    return mgL_to_nm(val) if target_unit=='nM' else val*1000.0
try:
    if mode!='Average inhibition': required=target/final_metric_for_dose(1.0)
    else: required=brentq(lambda d:final_metric_for_dose(d)-target,1e-6,1e7)
    st.success(f'Required dose per administration: {required:.1f} mg ({regimen})')
    st.caption(f'Target uses free {target_site.lower()} exposure in the final complete 24-hour window displayed.')
except Exception:
    st.warning('The requested target could not be solved within the prototype dose range.')

with st.expander('Model assumptions'):
    st.write('One compartment; first-order oral absorption and elimination; linear PK; dose-proportional exposure; Hill slope = 1; inhibition = C/(C+IC50).')
    st.write('Free plasma = total plasma × fu,p. Free brain = free plasma × Kp,uu. fu,p and Kp,uu are constant over time and dose.')
    st.write('The tool simulates exposure from assumed human PK parameters; it does not independently predict human PK or clinical dose.')
