"""Prespecified B0-B7 auction feature experiments; stage-locked auction-level CV."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/auction_feature_experiments"
STAGES=[.25,.50,.75]
SEEDS=[5910,5911,5912]
ITEMS={"xbox":"Xbox game console","pda":"Palm Pilot M515 PDA","watch":"Cartier wristwatch"}
OPEN=["openbid","duration"]
PRICE=["current_max_bid","price_lift","has_bid"]
COMP=["bid_count","unique_bidders","bids_per_bidder","bidder_hhi","bidder_missing_count"]
TIME=["first_bid_fraction","inactivity_fraction","recent_bid_count","recent_active_bidders","recent_new_bidders","recent_new_share","recent_bid_share","hhi_change","recent_max_gain"]
REPUTATION=["rating_median","low_rating_share","rating_missing_share"]
INTERACTIONS=["new_bidders_x_lift","hhi_change_x_lift"]
BASE=OPEN+PRICE
SETS={"B1_opening":OPEN,"B2_price":BASE,"B3_competition":BASE+COMP,"B4_dynamics":BASE+TIME,"B5_reputation":BASE+REPUTATION,"B6_full":BASE+COMP+TIME+REPUTATION,"A1_without_competition":BASE+TIME+REPUTATION,"A2_without_dynamics":BASE+COMP+REPUTATION,"A3_without_reputation":BASE+COMP+TIME,"B7_interactions":BASE+TIME+INTERACTIONS}

def hhi(frame):
    c=frame.bidder.dropna().value_counts()
    return float((c/c.sum()).pow(2).sum()) if len(c) else 0.

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/"data/raw/online_auctions/auction.csv"
    original=pd.read_csv(path)
    raw=original.copy()
    raw["duration"]=raw.auction_type.str.extract(r"(\d+)")[0].astype(int)
    status=raw.groupby("auctionid")[["price","openbid","duration","item"]].nunique()
    bad=status.index[status.gt(1).any(axis=1)]
    raw=raw.loc[~raw.auctionid.isin(bad)].copy()
    assert raw.bidtime.ge(0).all() and raw.bidtime.le(raw.duration).all()
    assert raw[["bid","openbid","price","bidtime"]].notna().all().all()
    assert raw.price.gt(0).all()
    records=[]
    for aid,g in raw.groupby("auctionid",sort=True):
        g=g.sort_values("bidtime",kind="stable")
        duration=int(g.duration.iloc[0])
        for stage in STAGES:
            cutoff=duration*stage
            start=cutoff-.1*duration
            prevstart=max(0,start-.1*duration)
            prefix=g.loc[g.bidtime.le(cutoff)]
            recent=prefix.loc[prefix.bidtime.gt(start)]
            earlier=prefix.loc[prefix.bidtime.le(start)]
            previous=prefix.loc[prefix.bidtime.gt(prevstart)&prefix.bidtime.le(start)]
            known=prefix.bidder.dropna()
            active=set(recent.bidder.dropna())
            new=active-set(earlier.bidder.dropna())
            initial=float(g.openbid.iloc[0])
            current=float(prefix.bid.max()) if len(prefix) else initial
            earliermax=float(earlier.bid.max()) if len(earlier) else initial
            ratings=prefix.loc[prefix.bidder.notna()].groupby("bidder").bidderrate.first()
            valid=ratings.dropna()
            row={"auctionid":str(aid),"item":g.item.iloc[0],"duration":duration,"stage":stage,"cutoff_days":cutoff,"observed_max_time":float(prefix.bidtime.max()) if len(prefix) else np.nan,"price":float(g.price.iloc[0]),"openbid":initial,"current_max_bid":current,"price_lift":current-initial,"has_bid":int(len(prefix)>0),"bid_count":len(prefix),"unique_bidders":known.nunique(),"bids_per_bidder":len(prefix)/max(known.nunique(),1),"bidder_hhi":hhi(prefix),"bidder_missing_count":int(prefix.bidder.isna().sum()),"first_bid_fraction":float(prefix.bidtime.min()/duration) if len(prefix) else np.nan,"inactivity_fraction":float((cutoff-prefix.bidtime.max())/duration) if len(prefix) else stage,"recent_bid_count":len(recent),"recent_active_bidders":len(active),"recent_new_bidders":len(new),"recent_new_share":len(new)/max(len(active),1),"recent_bid_share":len(recent)/max(len(prefix),1),"hhi_change":hhi(recent)-hhi(previous),"recent_max_gain":current-earliermax,"rating_median":float(valid.median()) if len(valid) else np.nan,"low_rating_share":float(valid.le(5).mean()) if len(valid) else np.nan,"rating_missing_share":float(ratings.isna().mean()) if len(ratings) else np.nan}
            row["new_bidders_x_lift"]=row["recent_new_bidders"]*row["price_lift"]
            row["hhi_change_x_lift"]=row["hhi_change"]*row["price_lift"]
            assert row["recent_new_bidders"]<=row["recent_active_bidders"]<=row["unique_bidders"]
            if len(prefix):assert row["observed_max_time"]<=cutoff
            records.append(row)
    features=pd.DataFrame(records)
    features.to_csv(OUT/"prefix_features.csv",index=False)
    splits=[]
    for key,item in ITEMS.items():
        f=features.loc[features.item.eq(item)&features.stage.eq(.5)].sort_values("auctionid").reset_index(drop=True)
        for seed in SEEDS:
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for fold,(train,test) in enumerate(cv.split(f,f.duration)):
                assert set(f.iloc[train].auctionid).isdisjoint(f.iloc[test].auctionid)
                for i in test:splits.append({"item_key":key,"auctionid":f.loc[i,"auctionid"],"seed":seed,"fold":fold})
    pd.DataFrame(splits).to_csv(OUT/"fold_assignments.csv",index=False)
    audit={"date":"2026-10-20","input_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"input_bid_rows":len(original),"input_auctions":int(original.auctionid.nunique()),"excluded_inconsistent_auction_ids":[str(v) for v in bad],"remaining_auctions":int(raw.auctionid.nunique()),"remaining_bid_rows":len(raw),"bid_above_final_retained":int(raw.bid.gt(raw.price).sum()),"primary_stage":.5,"supplementary_stages":[.25,.75],"seeds":SEEDS,"outer_cv":"3 repeats of duration-stratified 5-fold; same auction assignments across feature groups and stages; items modeled separately","parameters":{"ridge_alpha":10,"forest_trees":100,"forest_max_depth":5,"forest_min_leaf":5,"forest_seed":5910},"tuning":"None; fixed settings. B7 exploratory, not used for claims of confirmed significance.","definitions":{"recent_new_bidders":"Unique known bidders in (cutoff-0.1*duration, cutoff] absent from all bids at or before window start","recent_active_bidders":"All unique known bidders in recent window; distinct from truly new bidders","hhi_change":"Recent-window bidder HHI minus preceding adjacent window bidder HHI; empty window HHI defined as zero, interpret together with counts","price_lift":"Maximum recorded prefix bid minus opening bid; not necessarily public standing price","rating_missing_share":"Among identified prefix bidders; missing bidder identities separately counted"},"limitations":["No calendar timestamps: not forward chronological validation","Ratings may be collection-time snapshots; historical availability unverified","Same bidder can occur in different auctions; no bidder-disjoint validation","Repeated CV and exploratory bootstrap not independent final confirmation","No fraud labels or causal identification; price/currency provenance unverified","Earlier analyses already examined this dataset"],"versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    dictionary=[]
    for exp,cols in SETS.items():
        for col in cols:dictionary.append({"experiment":exp,"feature":col})
    pd.DataFrame(dictionary).to_csv(OUT/"experiment_features.csv",index=False)
    # Descriptive single-feature checks. No outcome-guided feature selection.
    associations=[]
    for (item,stage),g in features.groupby(["item","stage"]):
        for col in list(dict.fromkeys(BASE+COMP+TIME+REPUTATION+INTERACTIONS)):
            v=g[[col,"price"]].dropna()
            if len(v)>2 and v[col].nunique()>1:
                r=spearmanr(v[col],v.price)
                associations.append({"item":item,"stage":stage,"feature":col,"n":len(v),"spearman":r.statistic,"p_value":r.pvalue,"missing":int(g[col].isna().sum())})
            else:associations.append({"item":item,"stage":stage,"feature":col,"n":len(v),"spearman":np.nan,"p_value":np.nan,"missing":int(g[col].isna().sum())})
    assoc=pd.DataFrame(associations);assoc["bh_q_value"]=np.nan
    for _,idx in assoc.groupby(["item","stage"]).groups.items():
        valid=assoc.loc[idx,"p_value"].dropna()
        o=valid.sort_values()
        adj=np.minimum.accumulate((o.to_numpy()*len(o)/np.arange(1,len(o)+1))[::-1])[::-1]
        assoc.loc[o.index,"bh_q_value"]=np.minimum(adj,1)
    assoc.to_csv(OUT/"single_feature_associations.csv",index=False)
    for key,item in ITEMS.items():
        g=features.loc[features.item.eq(item)&features.stage.eq(.5)]
        fig,axs=plt.subplots(2,3,figsize=(13,8))
        for ax,col in zip(axs.flat,["current_max_bid","recent_new_bidders","bidder_hhi","hhi_change","recent_max_gain","rating_median"]):
            ax.scatter(g[col],g.price,s=18,alpha=.55)
            ax.set(xlabel=col,ylabel="Final price")
        fig.suptitle(item+" / 50% elapsed: descriptive associations")
        fig.tight_layout();fig.savefig(OUT/(key+"_single_features.png"),dpi=150);plt.close(fig)
    print("Prepared",len(features),"prefix rows,",raw.auctionid.nunique(),"auctions; true new-bidder definition checked.",flush=True)

def run_item(key):
    item=ITEMS[key]
    allf=pd.read_csv(OUT/"prefix_features.csv",dtype={"auctionid":str})
    splits=pd.read_csv(OUT/"fold_assignments.csv",dtype={"auctionid":str})
    splits=splits.loc[splits.item_key.eq(key)]
    predictions=[]
    for stage in STAGES:
        f=allf.loc[allf.item.eq(item)&allf.stage.eq(stage)].sort_values("auctionid").reset_index(drop=True)
        for seed in SEEDS:
            assignments=splits.loc[splits.seed.eq(seed)].set_index("auctionid").fold
            folds=f.auctionid.map(assignments)
            assert folds.notna().all()
            for fold in range(5):
                tr=f.loc[folds.ne(fold)];te=f.loc[folds.eq(fold)]
                assert set(tr.auctionid).isdisjoint(te.auctionid)
                def save(model,exp,p):
                    for aid,y,pred in zip(te.auctionid,te.price,p):
                        predictions.append({"item_key":key,"item":item,"stage":stage,"seed":seed,"fold":fold,"auctionid":aid,"actual":y,"model":model,"experiment":exp,"prediction":float(pred)})
                save("baseline","B0_median",np.full(len(te),tr.price.median()))
                save("baseline","B0_current_bid",te.current_max_bid.to_numpy())
                for exp,cols in SETS.items():
                    for family in ["ridge","random_forest"]:
                        learner=Ridge(alpha=10) if family=="ridge" else RandomForestRegressor(n_estimators=100,max_depth=5,min_samples_leaf=5,random_state=5910,n_jobs=1)
                        model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),learner)
                        model.fit(tr[cols],tr.price)
                        save(family,exp,model.predict(te[cols]))
        print(key,"completed stage",stage,flush=True)
    pd.DataFrame(predictions).to_csv(OUT/(key+"_predictions.csv"),index=False)

def finish():
    pred=pd.concat([pd.read_csv(OUT/(key+"_predictions.csv"),dtype={"auctionid":str}) for key in ITEMS],ignore_index=True)
    assert not pred.duplicated(["item_key","stage","seed","model","experiment","auctionid"]).any()
    metrics=[]
    for keys,g in pred.groupby(["item_key","stage","seed","model","experiment"]):
        metrics.append(dict(zip(["item_key","stage","seed","model","experiment"],keys),n=len(g),mae=mean_absolute_error(g.actual,g.prediction),rmse=np.sqrt(mean_squared_error(g.actual,g.prediction)),r2=r2_score(g.actual,g.prediction)))
    metrics=pd.DataFrame(metrics)
    summary=metrics.groupby(["item_key","stage","model","experiment"]).agg(n=("n","first"),mae=("mae","mean"),repeat_std=("mae","std"),rmse=("rmse","mean"),r2=("r2","mean")).reset_index()
    metrics.to_csv(OUT/"metrics_by_repeat.csv",index=False)
    summary.to_csv(OUT/"metrics_summary.csv",index=False)
    # Bootstrap paired per-auction errors, averaged over repeats; exploratory intervals.
    pred["abs_error"]=np.abs(pred.actual-pred.prediction)
    comparisons=[];rng=np.random.default_rng(5910)
    for (key,stage,family),g in pred.loc[pred.model.ne("baseline")].groupby(["item_key","stage","model"]):
        errors=g.groupby(["auctionid","experiment"]).abs_error.mean().unstack()
        pairs=[("B1_opening","B2_price")]+[("B2_price",s) for s in SETS if s not in ["B1_opening","B2_price"]]+[(s,"B6_full") for s in SETS if s.startswith("A")]
        for reference,new in pairs:
            gain=(errors[reference]-errors[new]).to_numpy()
            ids=rng.integers(0,len(gain),size=(2000,len(gain)))
            draws=gain[ids].mean(axis=1)
            pivot=metrics.loc[metrics.item_key.eq(key)&metrics.stage.eq(stage)&metrics.model.eq(family)].pivot(index="seed",columns="experiment",values="mae")
            comparisons.append({"item_key":key,"stage":stage,"model":family,"reference":reference,"new":new,"mae_gain":gain.mean(),"gain_percent":gain.mean()/errors[reference].mean()*100,"ci_low":np.quantile(draws,.025),"ci_high":np.quantile(draws,.975),"positive_repeats":int((pivot[reference]-pivot[new]).gt(0).sum())})
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(OUT/"paired_comparisons.csv",index=False)
    # Stratified tail sensitivity uses training-fold thresholds only; no refits/selection.
    tail=[]
    f=pd.read_csv(OUT/"prefix_features.csv",dtype={"auctionid":str})
    splits=pd.read_csv(OUT/"fold_assignments.csv",dtype={"auctionid":str})
    for (key,stage,seed,fold),g in pred.groupby(["item_key","stage","seed","fold"]):
        item=ITEMS[key]
        a=splits.loc[splits.item_key.eq(key)&splits.seed.eq(seed)]
        trainids=a.loc[a.fold.ne(fold),"auctionid"]
        trainprice=f.loc[f.item.eq(item)&f.stage.eq(stage)&f.auctionid.isin(trainids),"price"]
        threshold=trainprice.quantile(.95)
        for (family,exp),h in g.groupby(["model","experiment"]):
            for stratum,sub in [("up_to_train_p95",h.loc[h.actual.le(threshold)]),("above_train_p95",h.loc[h.actual.gt(threshold)])]:
                if len(sub):tail.append({"item_key":key,"stage":stage,"seed":seed,"fold":fold,"model":family,"experiment":exp,"stratum":stratum,"n":len(sub),"sum_abs_error":sub.abs_error.sum()})
    tail=pd.DataFrame(tail)
    tail=tail.groupby(["item_key","stage","model","experiment","stratum"]).agg(n_repeat_records=("n","sum"),sum_abs_error=("sum_abs_error","sum")).reset_index()
    tail["mae"]=tail.sum_abs_error/tail.n_repeat_records
    tail.to_csv(OUT/"tail_sensitivity.csv",index=False)
    for key in ITEMS:
        fig,axs=plt.subplots(1,2,figsize=(14,6))
        for ax,family in zip(axs,["ridge","random_forest"]):
            g=summary.loc[summary.item_key.eq(key)&summary.stage.eq(.5)&summary.model.eq(family)&summary.experiment.str.startswith("B")].sort_values("experiment")
            ax.barh(g.experiment,g.mae);ax.invert_yaxis()
            ax.set(title=key+" / "+family+" / 50%",xlabel="MAE (lower is better)")
        fig.tight_layout();fig.savefig(OUT/(key+"_B_sequence.png"),dpi=160);plt.close(fig)
    with pd.ExcelWriter(OUT/"auction_feature_experiments.xlsx",engine="openpyxl") as writer:
        for name,table in [("primary_50pct",summary.loc[summary.stage.eq(.5)]),("all_stages",summary),("paired_comparisons",comparison),("repeat_metrics",metrics),("single_features",pd.read_csv(OUT/"single_feature_associations.csv")),("tail_sensitivity",tail),("feature_dictionary",pd.read_csv(OUT/"experiment_features.csv")),("fold_assignments",splits)]:
            table.to_excel(writer,sheet_name=name,index=False)
    assert summary.n.groupby(summary.item_key).nunique().eq(1).all()
    print("Validated matched evaluation samples, unique predictions, and all reports.",flush=True)
    print(summary.loc[summary.stage.eq(.5)].to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["prepare","xbox","pda","watch","finish"])
    action=p.parse_args().action
    if action=="prepare":prepare()
    elif action=="finish":finish()
    else:run_item(action)
