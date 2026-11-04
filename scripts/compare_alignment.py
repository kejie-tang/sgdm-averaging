"""Compare corrected implementations with ef465cc, without neural training.

Run each phase separately; results are screening comparisons, not a claim that
the baseline commit exactly regenerates every raster embedded in the PDF.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/clt"),
               str(ROOT / "src/logistic")]
import sgdm_core as new
from clt_experiment import coverage_trial
from logistic_experiment import gradient

BASE = "ef465cc8507de36d67315c4ff7caca48bf23055c"
old = types.ModuleType("baseline")
exec(subprocess.check_output(
    ["git", "show", f"{BASE}:src/sgdm_core.py"], cwd=ROOT, text=True),
    old.__dict__)


def save(out, name, arrays, stats):
    np.savez_compressed(out / f"{name}.npz", **arrays)
    (out / f"{name}.json").write_text(json.dumps(stats, indent=2) + "\n")


def quadratic(args):
    sums = {}
    reflection = 0.
    for seed in range(args.trials):
        rng = np.random.RandomState(seed)
        V, b, u = old.draw_quadratic_problem(20000, 10, rng)
        before = old.build_quadratic_data(V, b, u)
        after = new.build_quadratic_data(V.transpose(0, 2, 1), b, u)
        schedule = old.make_batch_schedule(20000, 4000, 2000, seed, rng=rng)
        Aq, bq = old.batch_gradient_sequence(before["Ai"], b, 4000, schedule, 2000)
        settings = [(g, .001, 1., 10.) for g in
                    [0., .5, .7, .9, .96, .97, .98, .99]]
        for g, alpha, rho, shift in settings:
            t = old.run_sgdm(before["Ai"], b, before["x_star"], before["initial"],
                             4000, schedule, alpha, g, 2000, A_seq=Aq, b_seq=bq)
            tn = new.run_sgdm(after["Ai"], b, after["x_star"], after["initial"],
                              4000, schedule, alpha, g, 2000, A_seq=Aq, b_seq=bq)
            reflection = max(reflection, float(np.max(np.abs(t + tn))))
            for n0 in [250, 500, 750, 1000]:
                a = old.averaged_errors(t, before["x_star"], n0, 2000)**2
                z = new.averaged_errors(tn, after["x_star"], n0, 2000)**2
                key = f"g{g}_n{n0}"
                for suffix, curve in [("old", a), ("new", z)]:
                    k = f"{key}_{suffix}"
                    sums[k] = sums.get(k, 0.) + curve / args.trials
        if seed % 20 == 0:
            print(f"quadratic {seed}/{args.trials}", flush=True)
    stats = {"trials": args.trials, "N": 20000, "n": 2000,
             "max_reflection_error": reflection, "curves": {}}
    for k in sums:
        if not k.endswith("_old"):
            continue
        other = k[:-4] + "_new"
        a, b = sums[k], sums[other][:len(sums[k])]
        stats["curves"][k[:-4]] = {
            "max_relative_change": float(np.max(abs(b/a - 1))),
            "max_relative_change_after_100_averaged_points":
                float(np.max(abs(b[100:]/a[100:] - 1))),
            "final_relative_change": float(b[-1]/a[-1] - 1)}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (g, n0) in zip(axes.flat,
                          [(0.,250),(.96,250),(.99,500),(0.,500),(.96,500),(.99,1000)]):
        for suffix, ls in [("old","--"),("new","-")]:
            y = sums[f"g{g}_n{n0}_{suffix}"]
            ax.semilogy(np.arange(1,len(y)+1), y, ls, label=suffix)
        ax.set(title=f"gamma={g}, n0={n0}", xlabel="Averaged points",
               ylabel="Mean squared error")
        ax.legend()
    fig.suptitle(f"Averaging endpoint correction: {args.trials} paired data seeds")
    fig.savefig(args.out/"quadratic_comparison.png", dpi=160)
    plt.close(fig)
    save(args.out,"quadratic_comparison",sums,stats)


def logistic(args):
    sums = {}
    reflection = 0.
    # 141 stored iterates matches the published script's default horizon.
    n = 141
    for seed in range(args.trials):
        rng = np.random.RandomState(seed)
        X,y,theta = old.generate_logistic_data(20000,10,seed,rng=rng)
        schedule = old.make_batch_schedule(20000,4000,n,seed,rng=rng)
        xs=np.zeros((10,1))
        for _ in range(10*n):
            xs -= .5*gradient(xs,X,y)
        for g in [0.,.3,.5,.7,.8,.9]:
            paths={}
            for mode in ["old","new","scaled"]:
                x=np.zeros_like(xs);m=x.copy();traj=[]
                labels=y if mode=="old" else 1-y
                for k in range(n):
                    traj.append(x.copy())
                    ids=schedule[k*4000:(k+1)*4000]
                    grad=gradient(x,X[ids],labels[ids])
                    m=grad if k==0 and mode!="scaled" else g*m+(1-g)*grad
                    x=x-.5*m
                paths[mode]=np.array(traj)
            reflection=max(reflection,float(np.max(abs(paths["old"]+paths["new"]))))
            for mode in paths:
                target=xs if mode=="old" else -xs
                sums[f"g{g}_{mode}_last"]=sums.get(f"g{g}_{mode}_last",0)+np.sum(
                    (paths[mode]-target)**2,axis=(1,2))/args.trials
                for n0 in [10,40]:
                    fn=old.averaged_errors if mode=="old" else new.averaged_errors
                    a=fn(paths[mode],target,n0,99)**2
                    k=f"g{g}_{mode}_n{n0}"
                    sums[k]=sums.get(k,0)+a/args.trials
        if seed%20==0:
            print(f"logistic {seed}/{args.trials}",flush=True)
    stats={"trials":args.trials,"N":20000,"max_label_reflection_error":reflection}
    for g in [0.,.3,.5,.7,.8,.9]:
        for n0 in [10,40]:
            a,b=sums[f"g{g}_old_n{n0}"],sums[f"g{g}_new_n{n0}"]
            stats[f"g{g}_n{n0}"]={
                "max_relative_change":float(np.max(abs(b/a-1))),
                "final_relative_change":float(b[-1]/a[-1]-1)}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,kind in zip(axes,["last","n10","n40"]):
        for mode,ls in [("old","--"),("new","-"),("scaled",":")]:
            y=sums[f"g0.9_{mode}_{kind}"]
            ax.semilogy(np.arange(len(y)),y,ls,label=mode)
        ax.set(title=f"gamma=0.9, {kind}",xlabel="Plot index",ylabel="Mean squared error")
        ax.legend()
    fig.suptitle("old: original; new: labels/window fixed; scaled: standard first step")
    fig.savefig(args.out/"logistic_comparison.png",dpi=160);plt.close(fig)
    save(args.out,"logistic_comparison",sums,stats)


def clt(args):
    n,n0=1000,500
    rng=np.random.RandomState(1)
    V,b,u=old.draw_quadratic_problem(20000,10,rng)
    before=old.build_quadratic_data(V,b,u,shift=1.03,norm_const=1.04)
    after=new.build_quadratic_data(V.transpose(0,2,1),b,u,shift=1.0)
    phi=np.ones((10,1))/np.sqrt(10)
    if args.random_projection:
        after["phi"] = rng.rand(10,1)
        after["phi"] /= np.linalg.norm(after["phi"])
    cov=old.asymptotic_covariance(before["Ai"],b,before["x_star"],before["A"],
                                 4000,replace=False)
    exps=np.arange(-11,3);alphas=2.**exps;gammas=[0.,.8,.9]
    zb=np.full((args.trials,3,14),np.nan);za=zb.copy();zo=zb.copy()
    for seed in range(args.trials):
        sch=old.make_batch_schedule(20000,4000,n,seed)
        Aq,bq=old.batch_gradient_sequence(before["Ai"],b,4000,sch,n)
        with np.errstate(over="ignore",invalid="ignore"):
            for j,g in enumerate(gammas):
                for i,a in enumerate(alphas):
                    zb[seed,j,i],_=old.clt_projection(
                        before["Ai"],b,before["x_star"],before["initial"],
                        4000,sch,a,n,n0,cov,gamma=g,phi=phi,A_seq=Aq,b_seq=bq)
        sch_new=new.make_batch_schedule(20000,4000,n,seed,replace=True)
        za[seed]=coverage_trial(after,sch_new,4000,alphas,gammas,n,n0,"online",True)
        zo[seed]=coverage_trial(after,sch_new,4000,alphas,gammas,n,n0,"oracle",True)
        if seed%20==0:
            print(f"coverage {seed}/{args.trials}",flush=True)
    arrays={"exps":exps,"z_before":zb,"z_online":za,"z_oracle":zo,
            "phi_before":phi,"phi_after":after.get("phi",phi)}
    stats={"trials":args.trials,"N":20000,"n":n,"n0":n0,
           "binomial_standard_error_at_95pct":float(np.sqrt(.95*.05/args.trials)),
           "hessian_before":"(V.T@V+1.03I)/1.04",
           "hessian_after":"V.T@V+I",
           "projection_before":"ones/sqrt(d)",
           "projection_after":"seed-1 random" if args.random_projection else "ones/sqrt(d)",
           "note":"Baseline is current code at ef465cc, not extracted PDF figure values."}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for j,(ax,g) in enumerate(zip(axes,gammas)):
        for label,z,ls in [("baseline",zb,"--"),("online/iid",za,"-"),
                            ("oracle/iid",zo,":")]:
            y=(np.isfinite(z[:,j])&(abs(z[:,j])<=1.96)).mean(0)
            arrays[f"coverage_{label}_{g}"]=y
            ax.plot(exps,y,ls,marker=".",label=label)
        ax.axhline(.95,color="gray",lw=.8)
        ax.set(title=f"gamma={g}",xlabel="log2(alpha)",ylabel="Coverage",ylim=(-.02,1.02))
        ax.legend()
    extra="; new curves use a different fixed direction" if args.random_projection else ""
    fig.suptitle(f"Figure 12 comparisons: {args.trials} trajectories{extra}")
    fig.savefig(args.out/"coverage_comparison.png",dpi=160);plt.close(fig)
    save(args.out,"coverage_comparison",arrays,stats)


def histogram(args):
    n,n0=2000,1000
    rng=np.random.RandomState(1)
    V,b,u=old.draw_quadratic_problem(20000,10,rng)
    before=old.build_quadratic_data(V,b,u)
    after=new.build_quadratic_data(V.transpose(0,2,1),b,u)
    phi=rng.rand(10,1);phi/=np.linalg.norm(phi)
    cb=old.asymptotic_covariance(before['Ai'],b,before['x_star'],before['A'],4000,replace=False)
    ca=new.asymptotic_covariance(after['Ai'],b,after['x_star'],after['A'],4000,replace=True)
    zb=np.empty((args.trials,2));za=np.empty_like(zb)
    for seed in range(args.trials):
        for module,data,replace,cov,z in [(old,before,False,cb,zb),(new,after,True,ca,za)]:
            sch=module.make_batch_schedule(20000,4000,n,seed,replace=replace)
            Aq,bq=module.batch_gradient_sequence(data['Ai'],b,4000,sch,n)
            for j,gamma in enumerate([0.,.9]):
                z[seed,j],_=module.clt_projection(data['Ai'],b,data['x_star'],data['initial'],
                    4000,sch,.001,n,n0,cov,gamma=gamma,phi=phi,A_seq=Aq,b_seq=bq)
        if seed%20==0:print(f'histogram {seed}/{args.trials}',flush=True)
    stats={'trials':args.trials,'N':20000,'n':n,'n0':n0,
           'means_before':zb.mean(0).tolist(),'stds_before':zb.std(0).tolist(),
           'means_after':za.mean(0).tolist(),'stds_after':za.std(0).tolist()}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    x=np.linspace(-4,4,300)
    for j,ax in enumerate(axes):
        ax.hist(zb[:,j],bins=np.linspace(-4,4,21),density=True,histtype='step',label='baseline')
        ax.hist(za[:,j],bins=np.linspace(-4,4,21),density=True,histtype='step',label='new/iid')
        ax.plot(x,np.exp(-x*x/2)/np.sqrt(2*np.pi),'k:',label='N(0,1)')
        ax.set(title=['SGD','SGDM-0.9'][j],xlabel='Z',ylabel='Density');ax.legend()
    fig.suptitle(f'Figure 8 diagnostic: {args.trials} fixed-data trajectories')
    fig.savefig(args.out/'histogram_comparison.png',dpi=160);plt.close(fig)
    save(args.out,'histogram_comparison',{'z_before':zb,'z_after':za,'phi':phi},stats)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=["quadratic","logistic","clt","histogram"],required=True)
    ap.add_argument("--trials",type=int,default=200)
    ap.add_argument("--random_projection",action="store_true")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    {"quadratic":quadratic,"logistic":logistic,"clt":clt,"histogram":histogram}[args.phase](args)
