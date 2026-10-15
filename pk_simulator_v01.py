"""PK simulator v0.1: one-compartment oral PK, repeated dosing, QD/BID, dose targeting."""
import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title='Human PK Simulator', layout='wide')
st.title('Human PK Simulator')
st.caption('One-compartment oral PK with first-order absorption and elimination. Total plasma concentrations.')

with st.sidebar:
    st.header('Human PK inputs')
    cl_ml_min_kg = st.number_input('Plasma clearance (mL/min/kg)', min_value=0.001, value=5.0, step=0.1)
    vd_l_kg = st.number_input('Volume of distribution (L/kg)', min_value=0.001, value=1.0, step=0.1)
    f_pct = st.number_input('Bioavailability (%)', min_value=0.1, max_value=100.0, value=70.0, step=1.0)
    ka = st.number_input('ka (h^-1)', min_value=0.001, value=1.0, step=0.1)
    mw = st.number_input('Molecular weight (g/mol)', min_value=1.0, value=450.0, step=1.0)
    bw = st.number_input('Body weight (kg)', min_value=1.0, value=70.0, step=1.0)
    dose = st.number_input('Dose per administration (mg)', min_value=0.001, value=100.0, step=10.0)
    regimen = st.selectbox('Regimen', ['QD', 'BID'])
    days = st.slider('Simulation days', 1, 28, 7)
    ic50 = st.number_input('IC50 (nM)', min_value=0.001, value=100.0, step=10.0)

cl = cl_ml_min_kg * 60.0 / 1000.0 * bw  # L/h
vd = vd_l_kg * bw                        # L
kel = cl / vd
f = f_pct / 100.0
tau = 24.0 if regimen == 'QD' else 12.0

if abs(ka-kel) < 1e-8:
    st.error('ka is effectively equal to kel; choose a slightly different value for this prototype.')
    st.stop()


def single_c(t, dose_mg):
    """mg/L following a single oral dose at time 0."""
    t = np.asarray(t, dtype=float)
    return f * dose_mg * ka / (vd * (ka-kel)) * (np.exp(-kel*t)-np.exp(-ka*t))


def profile(dose_mg, total_h, dt=0.02):
    t = np.arange(0.0, total_h + dt/2, dt)
    c = np.zeros_like(t)
    for td in np.arange(0.0, total_h + 1e-9, tau):
        mask = t >= td
        c[mask] += single_c(t[mask]-td, dose_mg)
    return t, c


def mgL_to_nm(c):
    # mg/L == g/m3; mol/L = (mg/L)/1000 / MW; nM = mol/L*1e9
    return c * 1e6 / mw


def window_metrics(t, c, start, end):
    mask = (t >= start) & (t <= end + 1e-9)
    tw, cw = t[mask], c[mask]
    auc = np.trapezoid(cw, tw)
    cmax = float(np.max(cw))
    cavg = auc / (end-start)
    # Trough is concentration immediately before a scheduled dose; report minimum pre-dose within window.
    scheduled = np.arange(math.ceil(start/tau)*tau, end + 1e-9, tau)
    predose = []
    for td in scheduled:
        if td <= start + 1e-9:
            continue
        idx = np.searchsorted(t, td) - 1
        if idx >= 0:
            predose.append(c[idx])
    ctrough = float(min(predose)) if predose else float(cw[-1])
    inh = mgL_to_nm(cw) / (mgL_to_nm(cw) + ic50)
    avg_inh = float(np.trapezoid(inh, tw) / (end-start))
    return dict(Cmax=cmax, Ctrough=ctrough, Cavg=cavg, AUC24=auc, AvgInh=avg_inh)


total_h = max(24.0, days*24.0)
t, c = profile(dose, total_h)
first = window_metrics(t, c, 0.0, 24.0)
last_start = max(0.0, total_h - 24.0)
last = window_metrics(t, c, last_start, total_h)

st.subheader('Concentration-time profile')
unit = st.radio('Concentration units', ['nM', 'ng/mL'], horizontal=True)
y = mgL_to_nm(c) if unit == 'nM' else c * 1000.0
fig = go.Figure(go.Scatter(x=t/24.0, y=y, mode='lines', name=regimen))
fig.update_layout(xaxis_title='Time (days)', yaxis_title=f'Plasma concentration ({unit})', hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)


def display_metrics(m, label):
    conv = (lambda x: mgL_to_nm(x)) if unit == 'nM' else (lambda x: x*1000.0)
    auc_conv = mgL_to_nm(m['AUC24']) if unit == 'nM' else m['AUC24']*1000.0
    st.markdown(f'**{label}**')
    a,b,c1,d,e = st.columns(5)
    a.metric('Cmax', f"{conv(m['Cmax']):.2f} {unit}")
    b.metric('Ctrough', f"{conv(m['Ctrough']):.2f} {unit}")
    c1.metric('Cavg', f"{conv(m['Cavg']):.2f} {unit}")
    d.metric('AUC0-24', f"{auc_conv:.2f} {unit}·h")
    e.metric('Avg inhibition', f"{100*m['AvgInh']:.1f}%")

display_metrics(first, 'First 24 hours')
display_metrics(last, 'Final complete 24-hour window')

st.divider()
st.subheader('Dose calculator')
mode = st.selectbox('Dose target', ['Ctrough', 'Cavg', 'Cmax', 'Average inhibition'])
if mode == 'Average inhibition':
    target = st.number_input('Target average inhibition (%)', min_value=0.1, max_value=99.9, value=80.0)
else:
    target_unit = st.selectbox('Target units', ['nM', 'ng/mL'])
    target = st.number_input(f'Target {mode} ({target_unit})', min_value=0.001, value=100.0)


def final_metric_for_dose(d):
    tt, cc = profile(d, total_h)
    mm = window_metrics(tt, cc, last_start, total_h)
    if mode == 'Average inhibition':
        return 100*mm['AvgInh']
    val = mm[mode]
    return mgL_to_nm(val) if target_unit == 'nM' else val*1000.0

try:
    if mode != 'Average inhibition':
        base = final_metric_for_dose(1.0)
        required = target / base
    else:
        fn = lambda d: final_metric_for_dose(d)-target
        required = brentq(fn, 1e-6, 1e7)
    st.success(f'Required dose per administration: {required:.2f} mg ({regimen})')
    st.caption('Target is evaluated over the final complete 24-hour window displayed.')
except Exception:
    st.warning('The requested target could not be solved within the prototype dose range.')

with st.expander('Model assumptions'):
    st.write('One compartment; first-order oral absorption and elimination; linear PK; dose-proportional exposure; total plasma concentration; Hill slope = 1; inhibition = C/(C+IC50).')
    st.write('This tool simulates exposure from assumed human PK parameters. It does not independently predict human clearance, volume, bioavailability or clinical dose.')
