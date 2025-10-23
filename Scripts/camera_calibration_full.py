# 3-Stage Flame Tube Calibration with Mirrors (clean staged residuals)

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

CSV_PATH = "Data/blob_bounding_boxes.csv"
OUT_JSON = "Data/calibration_staged_clean.json"
TUBE_DIAM = 3.0  # inches

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
df["label"] = df["label"].astype(str).str.strip().str.upper()
df = df[df["label"].isin(["L","C","R"])]

def triplets(_df: pd.DataFrame):
    if "frame" in _df.columns:
        piv = _df.pivot_table(index="frame", columns="label",
                              values=["w","h","cx","cy"], aggfunc="median").dropna()
        piv.columns = [f"{a}_{b}" for a,b in piv.columns.to_flat_index()]
        return piv.reset_index(drop=True)
    seq = {k: v.reset_index(drop=True) for k,v in _df.groupby("label")}
    n = min(len(seq.get(k,[])) for k in ("L","C","R"))
    return pd.DataFrame({f"{c}_{L}": seq[L].loc[:n-1,c].to_numpy()
                         for L in ("L","C","R") for c in ("w","h","cx","cy")})

data = triplets(df)
nF = len(data)

# Panel centers for principal points
def panel_center(_df, L):
    sub = _df[_df["label"]==L]
    cx = 0.5*(float(np.nanmin(sub["cx"])) + float(np.nanmax(sub["cx"])))
    cy = 0.5*(float(np.nanmin(sub["cy"])) + float(np.nanmax(sub["cy"])))
    return cx, cy
ppC0 = panel_center(df, "C")
ppL0 = panel_center(df, "L")
ppR0 = panel_center(df, "R")

# -----------------------
# Stage 1: analytic distances
# -----------------------
wC = float(np.median(data["w_C"]))
wL = float(np.median(data["w_L"]))
wR = float(np.median(data["w_R"]))
ratio = 0.5*((wL/wC)+(wR/wC))

Dc_candidates = np.linspace(12, 36, 100)
Dm_candidates = Dc_candidates*(1-ratio)/(2*ratio)
mask = (Dm_candidates>=6) & (Dm_candidates<=20)

if np.any(mask):
    idx = np.argmin(np.abs(Dm_candidates[mask]-12))
    Dcam_analytic = Dc_candidates[mask][idx]
    Dmir_analytic = Dm_candidates[mask][idx]
else:
    Dcam_analytic, Dmir_analytic = 20.0, 10.0

f_fixed = 1700.0

print("Stage 1 analytic:")
print(f"  Dcam ≈ {Dcam_analytic:.2f} in")
print(f"  Dmir ≈ {Dmir_analytic:.2f} in")
print(f"  f    ≈ {f_fixed:.1f} px (fixed)")

# -----------------------
# Geometry helpers
# -----------------------
def ray_dir_from_pixel(u,v,f,cx,cy,k1=0.0):
    xn=(u-cx)/f; yn=(v-cy)/f
    if k1!=0:
        r2=xn*xn+yn*yn
        s=1+k1*r2
        xn*=s; yn*=s
    d=np.array([1,yn,xn])
    return d/np.linalg.norm(d)

def rot_yaw(yaw_rad):
    c,s=math.cos(yaw_rad),math.sin(yaw_rad)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def hinge_pitch_local(yaw_deg,pitch_deg,side):
    yaw=math.radians(yaw_deg*(1 if side=="R" else -1))
    pit=math.radians(pitch_deg)
    n=rot_yaw(yaw)@np.array([1,0,0])
    k=np.cross([0,1,0],n)
    if np.linalg.norm(k)<1e-9: k=[0,0,1]
    k=np.array(k)/np.linalg.norm(k)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Rk=np.eye(3)+math.sin(pit)*K+(1-math.cos(pit))*(K@K)
    return (Rk@n)/np.linalg.norm(Rk@n)

def mirror_plane(Dmir,yaw,pit,side):
    return np.array([-abs(Dmir),0,0]), hinge_pitch_local(yaw,pit,side)

def reflect_point(P,p0,n): return P-2*((P-p0)@n)*n
def reflect_dir(d,n): return d-2*np.dot(d,n)*n

def real_axis_point_dir(Dcam,Y):
    return np.array([Dcam,Y,0]), np.array([0,1,0])

def virtual_axis(Dcam,Y,Dmir,yaw,pit,side):
    A0_real,a_real=real_axis_point_dir(Dcam,Y)
    p0,n=mirror_plane(Dmir,yaw,pit,side)
    return reflect_point(A0_real,p0,n), reflect_dir(a_real,n)

def ray_line_closest(O,d,A0,a):
    w0=O-A0; ad=float(a@d); denom=1-ad*ad
    if denom<1e-12: t=0; s=float(a@(O-A0))
    else:
        t=(ad*(a@w0)-d@w0)/denom
        s=(a@w0-ad*(d@w0))/denom
    P_ray=O+t*d; P_axis=A0+s*a
    return float(np.linalg.norm(P_ray-P_axis)), P_axis

def width_from_axis_point(P_axis,f):
    X=float(P_axis[0])
    return (f*TUBE_DIAM)/X if X>0 else np.nan

def obs_pixels(i,view):
    cx=float(data[f"cx_{view}"].iloc[i])
    cy=float(data[f"cy_{view}"].iloc[i])
    h=float(data[f"h_{view}"].iloc[i])
    w=float(data[f"w_{view}"].iloc[i])
    return {"top":(cx,cy-0.5*h,w),
            "center":(cx,cy,w),
            "bottom":(cx,cy+0.5*h,w)}

def relaxed_residual(dist, tol=5.0):
    return max(0.0, dist - tol)

# -----------------------
# Residual builder
# -----------------------
def compute_residuals(yawL,yawR,pitL,pitR,k1,dvL,dvR,Yc,H):
    f,Dcam,Dmir=f_fixed,Dcam_analytic,Dmir_analytic
    r=[]; O=np.zeros(3)
    for i in range(nF):
        Yc_i=Yc[i]; Hi=H[i]; Yt=Yc_i+0.5*Hi; Yb=Yc_i-0.5*Hi
        # Center
        A0c,aC=real_axis_point_dir(Dcam,Yc_i)
        for key,Yp in (("top",Yt),("center",Yc_i),("bottom",Yb)):
            A0,_=real_axis_point_dir(Dcam,Yp)
            u,v,wobs=obs_pixels(i,"C")[key]
            d=ray_dir_from_pixel(u,v,f,*ppC0,k1)
            dist,P_axis=ray_line_closest(O,d,A0,aC)
            r.append(relaxed_residual(dist))
            if key=="center":
                wpred=width_from_axis_point(P_axis,f)
                r.append((wpred-wobs) if np.isfinite(wpred) else 1e3)
        # Left
        for key,Yp in (("top",Yt),("center",Yc_i),("bottom",Yb)):
            A0v,aL=virtual_axis(Dcam,Yp,Dmir,yawL,pitL,"L")
            u,v,wobs=obs_pixels(i,"L")[key]
            d=ray_dir_from_pixel(u,v+dvL,f,*ppL0,k1)
            dist,P_axis=ray_line_closest(O,d,A0v,aL)
            r.append(relaxed_residual(dist))
            if key=="center":
                wpred=width_from_axis_point(P_axis,f)
                r.append((wpred-wobs) if np.isfinite(wpred) else 1e3)
        # Right
        for key,Yp in (("top",Yt),("center",Yc_i),("bottom",Yb)):
            A0v,aR=virtual_axis(Dcam,Yp,Dmir,yawR,pitR,"R")
            u,v,wobs=obs_pixels(i,"R")[key]
            d=ray_dir_from_pixel(u,v+dvR,f,*ppR0,k1)
            dist,P_axis=ray_line_closest(O,d,A0v,aR)
            r.append(relaxed_residual(dist))
            if key=="center":
                wpred=width_from_axis_point(P_axis,f)
                r.append((wpred-wobs) if np.isfinite(wpred) else 1e3)
    return np.asarray(r)

# -----------------------
# Stage 1: yaw only
# -----------------------
Yc_seed=np.zeros(nF)
H_seed=np.full(nF,5.0)

def residuals_stage1(x):
    return compute_residuals(x[0],x[1],0,0,0,0,0,Yc_seed,H_seed)

x0=[70,70]; lb=[20,20]; ub=[85,85]
res1=least_squares(residuals_stage1,x0,bounds=(lb,ub),verbose=2,max_nfev=2000)
yawL,yawR=res1.x
print("Stage1 yaw:",yawL,yawR)

# -----------------------
# Stage 2: yaw+pitch+offsets+k1
# -----------------------
def residuals_stage2(x):
    yawL,yawR,pitL,pitR,k1,dvL,dvR=x
    return compute_residuals(yawL,yawR,pitL,pitR,k1,dvL,dvR,Yc_seed,H_seed)

x0 = [yawL,yawR,0,0,0,0,0]
lb = [20,20,-15,-15,-0.3,-50,-50]
ub = [90,90, 15, 15,  0.3, 50, 50]

res2 = least_squares(residuals_stage2, x0,
                     bounds=(lb,ub),
                     verbose=2, max_nfev=4000)
yawL,yawR,pitL,pitR,k1,dvL,dvR=res2.x
print("Stage2:",res2.x)

# -----------------------
# Stage 3: free Yc,H
# -----------------------
def residuals_stage3(x):
    n=len(x)//2
    Yc=x[:n]; H=x[n:]
    return compute_residuals(yawL,yawR,pitL,pitR,k1,dvL,dvR,Yc,H)

x0=np.concatenate([Yc_seed,H_seed])
lb=np.concatenate([np.full(nF,-100),np.full(nF,0)])
ub=np.concatenate([np.full(nF, 100),np.full(nF,100)])
res3=least_squares(residuals_stage3,x0,bounds=(lb,ub),verbose=2,max_nfev=4000)
Yc_sol=res3.x[:nF]; H_sol=res3.x[nF:]
print("Stage3 done.")

# -----------------------
# Summaries
# -----------------------
summary={
 "stage1":{"yawL":float(res1.x[0]),"yawR":float(res1.x[1])},
 "stage2":{"yawL":float(yawL),"yawR":float(yawR),
           "pitL":float(pitL),"pitR":float(pitR),
           "k1":float(k1),"dvL":float(dvL),"dvR":float(dvR)},
 "stage3":{"Yc_mean":float(np.mean(Yc_sol)),"H_mean":float(np.mean(H_sol))},
 "distances":{"Dcam_in":float(Dcam_analytic),"Dmir_in":float(Dmir_analytic),"f_px":f_fixed},
 "fit_quality":{"stage1_status":int(res1.status),
                "stage2_status":int(res2.status),
                "stage3_status":int(res3.status)}
}
Path(OUT_JSON).parent.mkdir(parents=True,exist_ok=True)
with open(OUT_JSON,"w") as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))

# -----------------------
# Visualize top view
# -----------------------
camera=np.array([Dcam_analytic,0])
mirrorL=np.array([-Dmir_analytic,-5])
mirrorR=np.array([-Dmir_analytic, 5])
fig,ax=plt.subplots(figsize=(8,6))
ax.plot([0,0],[-10,10],'k-',lw=3,label="Tube axis")
ax.scatter(*camera,c='blue',label="Camera")
ax.scatter(*mirrorL,c='red',label=f"L yaw={yawL:.1f}")
ax.scatter(*mirrorR,c='green',label=f"R yaw={yawR:.1f}")
ax.plot([camera[0],0],[camera[1],0],'b--',label="Direct")
ax.plot([camera[0],mirrorL[0]],[camera[1],mirrorL[1]],'r--')
ax.plot([mirrorL[0],0],[mirrorL[1],0],'r--',label="Left mirror")
ax.plot([camera[0],mirrorR[0]],[camera[1],mirrorR[1]],'g--')
ax.plot([mirrorR[0],0],[mirrorR[1],0],'g--',label="Right mirror")
ax.set_xlabel("X (in)")
ax.set_ylabel("Y (in)")
ax.legend(); ax.axis("equal")
ax.set_title("Approximate Top View Layout")
plt.show()
