import json, warnings, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":.3,"font.size":9,
                     "axes.spines.top":False,"axes.spines.right":False})
R = {}

# ---------- 1. Data ----------
df = pd.read_csv("aapl.csv", parse_dates=["Date"]).set_index("Date")
y = df["AAPL.Close"].rename("close")
D = y.index.copy()                      # trading dates (irregular: weekends/holidays missing)
y.index = pd.RangeIndex(len(y))        # integer index for statsmodels
R["n_obs"]=len(y); R["start"]=str(D.min().date()); R["end"]=str(D.max().date())
R["desc"]=y.describe().round(2).to_dict()
ly = np.log(y); r = ly.diff().dropna()
R["ret_stats"]={"mean_daily":r.mean(),"std_daily":r.std(),"skew":stats.skew(r),"exkurt":stats.kurtosis(r),
                "ann_vol":r.std()*np.sqrt(252)}

fig,ax=plt.subplots(2,1,figsize=(8,5.2),sharex=True)
ax[0].plot(D,y.values,color="#1f4e79",lw=1); ax[0].set_ylabel("Close (USD)"); ax[0].set_title("AAPL daily closing price, Feb 2015 – Feb 2017")
ax[1].plot(D[1:],r.values,color="#7f7f7f",lw=.7); ax[1].set_ylabel("Log return"); ax[1].set_title("Daily log returns")
plt.tight_layout(); plt.savefig("figs/01_series.png"); plt.close()

# ---------- 2. Stationarity ----------
def tests(s,reg="c"):
    a=adfuller(s.dropna(),autolag="AIC",regression=reg)
    k=kpss(s.dropna(),regression=reg,nlags="auto")
    return {"adf_stat":a[0],"adf_p":a[1],"adf_lags":a[2],"kpss_stat":k[0],"kpss_p":k[1]}
R["stationarity"]={"price":tests(y,"ct"),"log_price":tests(ly,"ct"),"log_return":tests(r,"c")}

fig,ax=plt.subplots(2,2,figsize=(8,5))
plot_acf(ly,lags=30,ax=ax[0,0],title="ACF – log price"); plot_pacf(ly,lags=30,ax=ax[0,1],title="PACF – log price",method="ywm")
plot_acf(r,lags=30,ax=ax[1,0],title="ACF – log return"); plot_pacf(r,lags=30,ax=ax[1,1],title="PACF – log return",method="ywm")
plt.tight_layout(); plt.savefig("figs/02_acf_pacf.png"); plt.close()

# ---------- 3. Split ----------
H=60
train,test = ly.iloc[:-H], ly.iloc[-H:]
R["split"]={"train_n":len(train),"test_n":len(test),"train_end":str(D[len(train)-1].date()),"test_start":str(D[len(train)].date())}

def metrics(actual_log, fc_log):
    a=np.exp(actual_log.values); f=np.exp(np.asarray(fc_log))
    e=a-f
    return {"RMSE":float(np.sqrt(np.mean(e**2))),"MAE":float(np.mean(np.abs(e))),
            "MAPE":float(np.mean(np.abs(e/a))*100)}

# ---------- 4. Model selection ----------
rows=[]
for p,q in itertools.product(range(4),range(4)):
    for trend in ["n","t"]:
        try:
            m=ARIMA(train,order=(p,1,q),trend=trend).fit()
            rows.append({"order":f"({p},1,{q})","trend":"drift" if trend=="t" else "none","AIC":m.aic,"BIC":m.bic,"p":p,"q":q,"tr":trend})
        except Exception as e: pass
grid=pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
R["grid_top"]=grid.head(8)[["order","trend","AIC","BIC"]].round(2).to_dict("records")
R["grid_best_bic"]=grid.sort_values("BIC").iloc[0][["order","trend","AIC","BIC"]].to_dict()
best=grid.iloc[0]; bo=(int(best.p),1,int(best.q)); bt=best.tr
R["best_order"]=list(bo); R["best_trend"]=bt
arima=ARIMA(train,order=bo,trend=bt).fit()
# parsimonious alternative picked by BIC
b2=grid.sort_values("BIC").iloc[0]; bo2=(int(b2.p),1,int(b2.q)); bt2=b2.tr
arima_b=ARIMA(train,order=bo2,trend=bt2).fit()
c=grid[(grid.p+grid.q)>=1].iloc[0]; oc=(int(c.p),1,int(c.q)); tc=c.tr
arima_c=ARIMA(train,order=oc,trend=tc).fit()
R["arima_c"]={"order":list(oc),"trend":tc,"AIC":float(arima_c.aic),"params":{k:float(v) for k,v in arima_c.params.items()},"pvalues":{k:float(v) for k,v in arima_c.pvalues.items()}}
R["arima_params"]={k:float(v) for k,v in arima.params.items()}
R["arima_pvalues"]={k:float(v) for k,v in arima.pvalues.items()}
R["arima_bic_model"]={"order":list(bo2),"trend":bt2}
R["arima_bic_params"]={k:float(v) for k,v in arima_b.params.items()}

# diagnostics
res=arima.resid.iloc[1:]
lb=acorr_ljungbox(res,lags=[10,20],return_df=True)
R["diag"]={"lb10_p":float(lb.lb_pvalue.iloc[0]),"lb20_p":float(lb.lb_pvalue.iloc[1]),
           "jb_p":float(stats.jarque_bera(res)[1]),"jb_stat":float(stats.jarque_bera(res)[0]),
           "arch_p":float(het_arch(res,nlags=10)[1]),"resid_std":float(res.std()),
           "lb20_sq_p":float(acorr_ljungbox(res**2,lags=[20],return_df=True).lb_pvalue.iloc[0])}
fig,ax=plt.subplots(2,2,figsize=(8,5))
ax[0,0].plot(D[res.index],res.values,lw=.7); ax[0,0].set_title("Residuals")
ax[0,1].hist(res,bins=30,density=True,alpha=.7); xs=np.linspace(res.min(),res.max(),200)
ax[0,1].plot(xs,stats.norm.pdf(xs,res.mean(),res.std()),"r"); ax[0,1].set_title("Residual histogram vs normal")
plot_acf(res,lags=30,ax=ax[1,0],title="Residual ACF")
stats.probplot(res,plot=ax[1,1]); ax[1,1].set_title("Q-Q plot")
plt.tight_layout(); plt.savefig("figs/03_diagnostics.png"); plt.close()

# ---------- 5. Multi-step forecasts on test window ----------
fcs={}
# naive
fcs["Naive (random walk)"]=np.repeat(train.iloc[-1],H)
# drift
drift=(train.iloc[-1]-train.iloc[0])/(len(train)-1)
fcs["Random walk + drift"]=train.iloc[-1]+drift*np.arange(1,H+1)
# SES on log price (no trend)
ses=SimpleExpSmoothing(train.values,initialization_method="estimated").fit()
fcs["Simple Exp. Smoothing"]=ses.forecast(H)
R["ses_alpha"]=float(ses.params["smoothing_level"])
# Holt (additive trend)
holt=ExponentialSmoothing(train.values,trend="add",initialization_method="estimated").fit()
fcs["Holt (additive trend)"]=holt.forecast(H)
R["holt_params"]={"alpha":float(holt.params["smoothing_level"]),"beta":float(holt.params["smoothing_trend"])}
# Damped Holt
hd=ExponentialSmoothing(train.values,trend="add",damped_trend=True,initialization_method="estimated").fit()
fcs["Damped Holt"]=hd.forecast(H)
R["damped_params"]={"alpha":float(hd.params["smoothing_level"]),"beta":float(hd.params["smoothing_trend"]),"phi":float(hd.params["damping_trend"])}
# ARIMA
fa=arima.get_forecast(H); fcs[f"ARIMA{bo} {'+drift' if bt=='t' else ''}".strip()]=fa.predicted_mean.values
fb=arima_b.get_forecast(H); name_b=f"ARIMA{bo2} {'+drift' if bt2=='t' else ''}".strip()
if name_b not in fcs: fcs[name_b]=fb.predicted_mean.values
fc_=arima_c.get_forecast(H); name_c=f"ARIMA{oc} {'+drift' if tc=='t' else ''}".strip(); fcs[name_c]=fc_.predicted_mean.values
names_arima=[k for k in fcs if k.startswith("ARIMA")]
R["multistep"]={k:metrics(test,v) for k,v in fcs.items()}
# direction of cumulative change
for k,v in fcs.items():
    R["multistep"][k]["dir_correct"]=bool(np.sign(v[-1]-train.iloc[-1])==np.sign(test.iloc[-1]-train.iloc[-1]))
# interval coverage for primary ARIMA
ci=fa.conf_int(alpha=.05).values
inside=((test.values>=ci[:,0])&(test.values<=ci[:,1])).mean()
R["arima_ci_coverage_60d"]=float(inside)

fig,ax=plt.subplots(figsize=(8.4,4.4))
hist=np.exp(ly.iloc[-H-120:-H]); ax.plot(D[-H-120:-H],hist.values,color="#444",lw=1,label="Train (last 120 days)")
TD=D[-H:]; ax.plot(TD,np.exp(test.values),color="black",lw=1.6,label="Actual (test)")
cols=["#d62728","#2ca02c","#9467bd","#ff7f0e","#8c564b","#1f77b4","#17becf"]
main=[k for k in fcs if k in ("Naive (random walk)",*names_arima,"Damped Holt","Holt (additive trend)")]
for c,k in zip(cols,main): ax.plot(TD,np.exp(fcs[k]),lw=1.2,ls="--",color=c,label=k)
ax.fill_between(TD,np.exp(ci[:,0]),np.exp(ci[:,1]),color="#1f77b4",alpha=.12,label="ARIMA 95% PI")
ax.set_title("60-day multi-step forecasts vs. actual (test window)"); ax.set_ylabel("USD"); ax.legend(fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig("figs/04_multistep.png"); plt.close()

# ---------- 6. Rolling-origin one-step-ahead (parameters fixed, state updated) ----------
def rolling_one_step(model_fn):
    preds=[]; 
    for i in range(H):
        hist=ly.iloc[:len(train)+i]
        preds.append(model_fn(hist))
    return np.array(preds)
def f_naive(h): return h.iloc[-1]
def f_drift(h): return h.iloc[-1]+(h.iloc[-1]-h.iloc[0])/(len(h)-1)
def f_ses(h):
    m=SimpleExpSmoothing(h.values,initialization_method="estimated").fit(); return m.forecast(1)[0]
def f_damped(h):
    m=ExponentialSmoothing(h.values,trend="add",damped_trend=True,initialization_method="estimated").fit(); return m.forecast(1)[0]
def mk_arima(order,tr):
    def f(h):
        m=ARIMA(h,order=order,trend=tr).fit(); return m.forecast(1).iloc[0]
    return f
roll={"Naive (random walk)":f_naive,"Random walk + drift":f_drift,"Simple Exp. Smoothing":f_ses,"Damped Holt":f_damped,
      f"ARIMA{bo} {'+drift' if bt=='t' else ''}".strip():mk_arima(bo,bt)}
if name_b not in roll: roll[name_b]=mk_arima(bo2,bt2)
roll[name_c]=mk_arima(oc,tc)
rp={k:rolling_one_step(f) for k,f in roll.items()}
R["rolling1"]={k:metrics(test,v) for k,v in rp.items()}
for k,v in rp.items():
    prev=np.r_[train.iloc[-1],test.values[:-1]]
    R["rolling1"][k]["dir_acc"]=None if np.allclose(v,prev) else float(np.mean(np.sign(v-prev)==np.sign(test.values-prev))*100)

# Diebold-Mariano (squared error, h=1) vs naive
def dm(e1,e2):
    d=e1**2-e2**2; n=len(d); s=d.mean()/np.sqrt(d.var(ddof=1)/n); return float(s),float(2*(1-stats.norm.cdf(abs(s))))
a=np.exp(test.values); en=a-np.exp(rp["Naive (random walk)"])
R["dm_vs_naive"]={k:(None if np.allclose(v,rp["Naive (random walk)"]) else dm(a-np.exp(v),en)) for k,v in rp.items() if k!="Naive (random walk)"}

fig,ax=plt.subplots(figsize=(8.4,3.8))
TD=D[-H:]; ax.plot(TD,np.exp(test.values),color="black",lw=1.6,label="Actual")
for c,k in zip(cols,[k for k in rp if k in ("Naive (random walk)",*names_arima,"Damped Holt")]):
    ax.plot(TD,np.exp(rp[k]),lw=1,ls="--",color=c,label=k)
ax.set_title("Rolling one-step-ahead forecasts (models re-estimated daily)"); ax.set_ylabel("USD"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/05_rolling.png"); plt.close()

# Error bar chart
fig,ax=plt.subplots(1,2,figsize=(8.4,3.4))
ks=list(R["multistep"].keys()); ax[0].barh(ks,[R["multistep"][k]["RMSE"] for k in ks],color="#1f4e79"); ax[0].set_title("60-day horizon RMSE (USD)")
ks2=list(R["rolling1"].keys()); ax[1].barh(ks2,[R["rolling1"][k]["RMSE"] for k in ks2],color="#2e8b57"); ax[1].set_title("1-step rolling RMSE (USD)")
for a_ in ax: a_.tick_params(axis="y",labelsize=7)
plt.tight_layout(); plt.savefig("figs/06_rmse.png"); plt.close()


# ---------- 6b. Volatility: GARCH(1,1) for prediction intervals ----------
from arch import arch_model
rr=(r*100)                                  # percent log returns, index 1..n-1
r_tr=rr.iloc[:len(train)-1]
gm=arch_model(r_tr,mean="Zero",vol="GARCH",p=1,q=1,dist="t").fit(disp="off")
R["garch"]={k:float(v) for k,v in gm.params.items()}; R["garch"]["persistence"]=float(gm.params["alpha[1]"]+gm.params["beta[1]"])
R["garch"]["loglik"]=float(gm.loglikelihood); R["garch"]["aic"]=float(gm.aic)
# 1-step-ahead rolling 95% intervals: constant-variance (train sigma) vs GARCH conditional variance (params fixed)
full=arch_model(rr,mean="Zero",vol="GARCH",p=1,q=1,dist="t").fix(gm.params)
fv=full.forecast(start=len(train)-1,horizon=1,reindex=False).variance.values.ravel()[:H]   # var of next-day return, %^2
sig_g=np.sqrt(fv)/100; sig_c=np.sqrt(arima.sigma2) if hasattr(arima,"sigma2") else res.std()
sig_c=float(res.std())
prev_l=np.r_[train.iloc[-1],test.values[:-1]]
z=1.96
cov_c=np.mean(np.abs(test.values-prev_l)<=z*sig_c); cov_g=np.mean(np.abs(test.values-prev_l)<=z*sig_g)
R["interval1"]={"sigma_const":sig_c,"sigma_garch_mean":float(sig_g.mean()),"sigma_garch_min":float(sig_g.min()),"sigma_garch_max":float(sig_g.max()),
                "cov_const":float(cov_c),"cov_garch":float(cov_g),
                "width_const_usd":float(np.mean(2*z*sig_c*np.exp(prev_l))),"width_garch_usd":float(np.mean(2*z*sig_g*np.exp(prev_l)))}
# 60-day horizon PI under GARCH
f60=gm.forecast(horizon=H,reindex=False).variance.values.ravel()/1e4
cum=np.sqrt(np.cumsum(f60)); lo=train.iloc[-1]-z*cum; hi=train.iloc[-1]+z*cum
R["interval60"]={"cov_garch":float(((test.values>=lo)&(test.values<=hi)).mean()),"cov_const":R["arima_ci_coverage_60d"],
   "width_end_garch":float(np.exp(hi[-1])-np.exp(lo[-1])),"width_end_const":float(np.exp(ci[-1,1])-np.exp(ci[-1,0]))}
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
cv=gm.conditional_volatility
ax[0].plot(D[cv.index],cv.values*np.sqrt(252),color="#7b3294",lw=.9); ax[0].set_title("GARCH(1,1) conditional vol (annualised %)"); ax[0].tick_params(labelsize=7)
ax[1].plot(TD,test.values-prev_l,color="black",lw=.7,label="Daily log return")
ax[1].fill_between(TD,-z*sig_g,z*sig_g,color="#7b3294",alpha=.2,label="GARCH 95%")
ax[1].axhline(z*sig_c,color="#1f77b4",ls="--",lw=1,label="Const-σ 95%"); ax[1].axhline(-z*sig_c,color="#1f77b4",ls="--",lw=1)
ax[1].set_title("1-step 95% intervals (test window)"); ax[1].legend(fontsize=7); ax[1].tick_params(labelsize=7)
plt.tight_layout(); plt.savefig("figs/09_garch.png"); plt.close()

# ---------- 7. Final model: refit on all data, 20-day-ahead forecast ----------
F=20
fm=ARIMA(ly,order=bo,trend=bt).fit(); ff=fm.get_forecast(F)
mu=ff.predicted_mean; ci80=ff.conf_int(alpha=.20); ci95=ff.conf_int(alpha=.05)
fidx=pd.bdate_range(D[-1]+pd.Timedelta(days=1),periods=F)
R["final"]={"params":{k:float(v) for k,v in fm.params.items()},"last_close":float(y.iloc[-1]),
  "fc_h1":float(np.exp(mu.iloc[0])),"fc_h20":float(np.exp(mu.iloc[-1])),
  "pi95_h20":[float(np.exp(ci95.iloc[-1,0])),float(np.exp(ci95.iloc[-1,1]))],
  "pi80_h20":[float(np.exp(ci80.iloc[-1,0])),float(np.exp(ci80.iloc[-1,1]))]}
fig,ax=plt.subplots(figsize=(8.4,4))
ax.plot(D[-150:],y.iloc[-150:].values,color="black",lw=1.2,label="Observed")
ax.plot(fidx,np.exp(mu.values),color="#d62728",lw=1.6,label="ARIMA forecast")
ax.fill_between(fidx,np.exp(ci95.iloc[:,0]),np.exp(ci95.iloc[:,1]),color="#d62728",alpha=.12,label="95% PI")
ax.fill_between(fidx,np.exp(ci80.iloc[:,0]),np.exp(ci80.iloc[:,1]),color="#d62728",alpha=.22,label="80% PI")
ax.set_title("20-trading-day-ahead forecast beyond the sample (refit on full data)"); ax.set_ylabel("USD"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/07_final.png"); plt.close()

# Stationarity before/after figure
fig,ax=plt.subplots(1,3,figsize=(9,2.8))
ax[0].plot(D,y.values,lw=.8); ax[0].set_title("Price (non-stationary)")
ax[1].plot(D,ly.values,lw=.8,color="#2ca02c"); ax[1].set_title("Log price")
ax[2].plot(D[1:],r.values,lw=.6,color="#d62728"); ax[2].set_title("Log return (Δ log price)")
for a_ in ax: a_.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig("figs/08_transform.png"); plt.close()

json.dump(R,open("results.json","w"),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))
