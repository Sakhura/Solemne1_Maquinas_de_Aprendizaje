
import sys
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")          # sin ventana: las figuras se guardan en figs/

def display(obj):
    """Reemplazo de IPython.display para ejecutar fuera de Jupyter."""
    if hasattr(obj, "data") and hasattr(obj, "format"):   # pandas Styler
        obj = obj.data
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(obj, end="\n\n")


import itertools, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
FIG = "figs/"
import os; os.makedirs(FIG, exist_ok=True)
DATA = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "esi-2023---personas.csv"



print("\n" + "=" * 70 + "\n1. Carga, descripción y limpieza (5 pts)\n" + "=" * 70)


raw = pd.read_csv(DATA, low_memory=False)
print(f"Filas: {raw.shape[0]:,} · Columnas: {raw.shape[1]}")
print("Tipos de dato:", raw.dtypes.value_counts().to_dict())
raw[["edad", "sexo", "cine", "region", "ocup_ref", "habituales", "ing_t_t", "fact_cal_esi"]].head()



import re
LEAK = [c for c in raw.columns if c.startswith(("ing_", "imp_"))
        or c in {"ss_t", "svar_t", "reg_t", "gan_t", "autoconsumo_t", "ahorroimp_t", "tim", "id_mediana"}
        or re.match(r"d\d.*_(monto|montob|promedio|promediob|porcentaje)$", c)]
print(f"{len(LEAK)} columnas descartadas por fuga, p. ej.:", LEAK[:10])

REGION = {1:"Tarapacá",2:"Antofagasta",3:"Atacama",4:"Coquimbo",5:"Valparaíso",6:"O'Higgins",7:"Maule",
          8:"Biobío",9:"Araucanía",10:"Los Lagos",11:"Aysén",12:"Magallanes",13:"Metropolitana",
          14:"Los Ríos",15:"Arica y Parinacota",16:"Ñuble"}
EDUC = {1:"1 Nunca estudió",3:"3 Primaria (1)",4:"4 Primaria (2)",5:"5 Secundaria",6:"6 Técnica superior",
        7:"7 Universitaria",8:"8 Postítulo/Magíster",9:"9 Doctorado"}
CATEG = {1:"Empleador",2:"Cuenta propia",3:"Asalariado privado",4:"Asalariado público",
         5:"Serv. doméstico p. afuera",6:"Serv. doméstico p. adentro",7:"Familiar no remunerado"}
OCUP = {1:"Directores/gerentes",2:"Profesionales",3:"Técnicos",4:"Apoyo administrativo",5:"Servicios y ventas",
        6:"Agrícolas calificados",7:"Oficios y artesanos",8:"Operadores de máquinas",9:"Ocup. elementales",10:"Otros no identificados"}

pob = raw[(raw["ocup_ref"] == 1)].copy()
print(f"Ocupados: {len(pob):,} · con ing_t_t = 0: {(pob.ing_t_t == 0).sum():,}")
pob = pob[pob["ing_t_t"] > 0]

df = pd.DataFrame({
    "ing":        pob["ing_t_t"],
    "edad":       pob["edad"].astype(float),
    "mujer":      (pob["sexo"] == 2).astype(int),
    "educ":       pob["cine"].replace({999: np.nan}).map(EDUC),
    "region":     pob["region"].map(REGION),
    "rural":      (pob["tipo"] == 3).astype(int),
    "horas":      pob["habituales"].replace({0: np.nan, 888: np.nan, 999: np.nan, 8888: np.nan, 9999: np.nan}),
    "categoria":  pob["categoria_ocupacion"].map(CATEG),
    "informal":   pob["ocup_form"].map({1: 0, 2: 1}),          # 999 = sin clasificación → NaN
    "ocupacion":  pob["b1"].replace({999: np.nan}).map(OCUP),
    "extranjero": (pob["nacionalidad"] != 152).astype(int),
    "w":          pob["fact_cal_esi"],
})
print(f"Población de análisis antes de limpiar: {len(df):,}")



na = df.isna().sum().to_frame("faltantes")
na["%"] = 100 * na["faltantes"] / len(df)
display(na[na.faltantes > 0])
n0 = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Se eliminan {n0 - len(df):,} filas ({100*(n0-len(df))/n0:.2f} %). Quedan {len(df):,}.")



num = ["ing", "edad", "horas"]
desc = df[num].describe(percentiles=[.1, .25, .5, .75, .9, .99]).T
desc["asimetría"] = df[num].skew()
display(desc)

def wq(x, w, q):
    o = np.argsort(x.values); cw = np.cumsum(w.values[o]); return x.values[o][np.searchsorted(cw, q * cw[-1])]
print(f"Mediana muestral del ingreso:   ${df.ing.median():>12,.0f}")
print(f"Mediana poblacional (factor):   ${wq(df.ing, df.w, .5):>12,.0f}")
print(f"Media muestral del ingreso:     ${df.ing.mean():>12,.0f}")



for c in ["mujer", "rural", "informal", "extranjero"]:
    print(f"{c:<11} proporción = {df[c].mean():.3f}")
for c in ["educ", "categoria", "ocupacion"]:
    t = df.groupby(c)["ing"].agg(n="size", mediana="median").sort_values("mediana")
    t["%"] = 100 * t["n"] / len(df)
    display(t.style.format({"mediana": "${:,.0f}", "%": "{:.1f}"}).set_caption(c))


fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(df.ing / 1e6, bins=120, color="#4C72B0"); ax[0].set_xlim(0, 5)
ax[0].set(title="ing_t_t (millones CLP, recortado a 5 M)", xlabel="millones de pesos", ylabel="frecuencia")
ax[1].hist(np.log(df.ing), bins=80, color="#55A868")
ax[1].set(title="log(ing_t_t)", xlabel="log pesos")
plt.tight_layout(); plt.savefig(FIG + "f1_distribucion.png", dpi=150); plt.show() if "ipykernel" in sys.modules else plt.close()
print(f"Asimetría: ingreso = {df.ing.skew():.1f} · log(ingreso) = {np.log(df.ing).skew():.2f}")



df.drop(columns="w").to_csv("ingresos_chile_2023.csv", index=False)
print("Exportado ingresos_chile_2023.csv:", df.shape)



print("\n" + "=" * 70 + "\n2. Modelos de regresión (15 pts)\n" + "=" * 70)



GROUPS_RAW = {
    "edad (cuadrática)": ["edad", "edad2"], "mujer": ["mujer"], "educ": None, "region": None,
    "rural": ["rural"], "horas": ["horas"], "categoria": None, "informal": ["informal"],
    "ocupacion": None, "extranjero": ["extranjero"],
}
d = df.copy(); d["edad2"] = d["edad"] ** 2
cat_cols = ["educ", "region", "categoria", "ocupacion"]
ref = {"educ": "5 Secundaria", "region": "Metropolitana", "categoria": "Asalariado privado", "ocupacion": "Ocup. elementales"}
for c in cat_cols:
    d[c] = pd.Categorical(d[c], categories=[ref[c]] + sorted(set(d[c]) - {ref[c]}))
X_all = pd.get_dummies(d[["edad", "edad2", "mujer", "rural", "horas", "informal", "extranjero"] + cat_cols],
                       columns=cat_cols, drop_first=True, dtype=float)
GROUPS = {g: (cols if cols else [c for c in X_all.columns if c.startswith(g + "_")]) for g, cols in GROUPS_RAW.items()}
y_all = np.log(df["ing"].values)
print("Predictores (columnas):", X_all.shape[1], "· grupos:", len(GROUPS))
print({g: len(c) for g, c in GROUPS.items()})

X_tr, X_te, y_tr, y_te, ing_tr, ing_te = train_test_split(
    X_all, y_all, df["ing"].values, test_size=0.20, random_state=SEED)
print(f"Entrenamiento: {len(X_tr):,} · Prueba (hold-out 20 %): {len(X_te):,}")
cols = list(X_all.columns)





def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
Xtr_np, ytr = X_tr.values, y_tr

simple, kfold = [], []
for s in range(30):
    a, b, ya, yb = train_test_split(Xtr_np, ytr, test_size=.2, random_state=s)
    simple.append(rmse(yb, LinearRegression().fit(a, ya).predict(b)))
    errs = []
    for tr, va in KFold(10, shuffle=True, random_state=s).split(Xtr_np):
        errs.append(rmse(ytr[va], LinearRegression().fit(Xtr_np[tr], ytr[tr]).predict(Xtr_np[va])))
    kfold.append(np.mean(errs))
base_rmse = rmse(ytr, np.full_like(ytr, ytr.mean()))
cmp = pd.DataFrame({"Validación simple": simple, "10-fold CV": kfold}).agg(["mean", "std", "min", "max"]).T
display(cmp)
print(f"Modelo nulo (predice la media): RMSE log = {base_rmse:.3f}")
print(f"La desviación de la estimación con validación simple es {cmp.loc['Validación simple','std']/cmp.loc['10-fold CV','std']:.1f} veces la de 10-fold.")




gnames = list(GROUPS)
gidx = {g: [cols.index(c) for c in GROUPS[g]] for g in gnames}
subsets = [s for k in range(1, len(gnames) + 1) for s in itertools.combinations(gnames, k)]
print("Subconjuntos evaluados por fold:", len(subsets))

def fit_ne(G, b, idx):
    return np.linalg.solve(G[np.ix_(idx, idx)], b[idx])

def add_const(A): return np.column_stack([np.ones(len(A)), A])
K_BS = 5
t0 = time.time()
fold_res = []   # (fold, k, subset, train_rmse, val_rmse)
all_cv = {s: [] for s in subsets}
for f, (tr, va) in enumerate(KFold(K_BS, shuffle=True, random_state=SEED).split(Xtr_np)):
    A, Av = add_const(Xtr_np[tr]), add_const(Xtr_np[va])
    G, b, yy = A.T @ A, A.T @ ytr[tr], ytr[tr] @ ytr[tr]
    for s in subsets:
        idx = [0] + [i + 1 for g in s for i in gidx[g]]
        beta = fit_ne(G, b, idx)
        rss = yy - 2 * beta @ b[idx] + beta @ G[np.ix_(idx, idx)] @ beta
        v = rmse(ytr[va], Av[:, idx] @ beta)
        fold_res.append((f, len(s), s, np.sqrt(rss / len(tr)), v)); all_cv[s].append(v)
print(f"Búsqueda exhaustiva: {len(fold_res):,} ajustes en {time.time()-t0:.1f} s")

fr = pd.DataFrame(fold_res, columns=["fold", "k", "subset", "train_rmse", "val_rmse"])
best_per_fold = fr.loc[fr.groupby(["fold", "k"])["train_rmse"].idxmin()]
bs_curve = best_per_fold.groupby("k").agg(cv=("val_rmse", "mean"), sd=("val_rmse", "std"), train=("train_rmse", "mean"))
bs_curve["se"] = bs_curve["sd"] / np.sqrt(K_BS)
k_min = bs_curve["cv"].idxmin()
umbral = bs_curve.loc[k_min, "cv"] + bs_curve.loc[k_min, "se"]
k_1se = int(bs_curve.index[bs_curve["cv"] <= umbral].min())
display(bs_curve)
print(f"k con mínimo CV = {k_min} · umbral 1-SE = {umbral:.4f} · k elegido (1-SE) = {k_1se}")



A_full = add_const(Xtr_np); G_full, b_full, yy_full = A_full.T @ A_full, A_full.T @ ytr, ytr @ ytr
best_full = {}
for s in subsets:
    idx = [0] + [i + 1 for g in s for i in gidx[g]]
    beta = fit_ne(G_full, b_full, idx)
    rss = yy_full - 2 * beta @ b_full[idx] + beta @ G_full[np.ix_(idx, idx)] @ beta
    if len(s) not in best_full or rss < best_full[len(s)][1]:
        best_full[len(s)] = (s, rss)
for k in range(1, len(gnames) + 1):
    print(f"k={k:>2}: {', '.join(best_full[k][0])}")
BS_SET = best_full[k_1se][0]
BS_COLS = [c for g in BS_SET for c in GROUPS[g]]
print(f"\nModelo best subsets elegido (k={k_1se}, {len(BS_COLS)} columnas): {BS_SET}")
print("Grupos descartados:", [g for g in gnames if g not in BS_SET])




fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(bs_curve.index, bs_curve.train, "o--", color="gray", label="Error de entrenamiento")
ax.errorbar(bs_curve.index, bs_curve.cv, yerr=bs_curve.se, fmt="o-", color="#C44E52", capsize=3, label="Error CV 5-fold ± 1 SE")
ax.axhline(umbral, ls=":", color="k", lw=1, label="umbral 1-SE")
ax.axvline(k_1se, color="#4C72B0", lw=1.2, label=f"k elegido = {k_1se}")
ax.set(xlabel="Número de grupos de predictores (k)", ylabel="RMSE (log ingreso)", title="Best subsets: error vs complejidad")
ax.legend(); plt.tight_layout(); plt.savefig(FIG + "f2_bestsubsets.png", dpi=150); plt.show() if "ipykernel" in sys.modules else plt.close()





K = 10
folds = list(KFold(K, shuffle=True, random_state=SEED).split(Xtr_np))
alphas_ridge = np.logspace(-2, 5, 60)
alphas_lasso = np.logspace(0, -4, 60)          # orden decreciente (requisito de lasso_path)

def cv_curve(model, alphas):
    cv = np.zeros((K, len(alphas))); tr_err = np.zeros((K, len(alphas))); nz = np.zeros((K, len(alphas)))
    for f, (tr, va) in enumerate(folds):
        sc = StandardScaler().fit(Xtr_np[tr]); A, Av = sc.transform(Xtr_np[tr]), sc.transform(Xtr_np[va])
        if model == "lasso":
            al_ret, coefs, _ = lasso_path(A, ytr[tr] - ytr[tr].mean(), alphas=alphas)
            assert np.allclose(al_ret, alphas)          # lasso_path exige/devuelve la grilla en orden decreciente
            for j in range(len(alphas)):
                c = coefs[:, j]; mu = ytr[tr].mean()
                cv[f, j] = rmse(ytr[va], mu + Av @ c); tr_err[f, j] = rmse(ytr[tr], mu + A @ c); nz[f, j] = (c != 0).sum()
        else:
            for j, a in enumerate(alphas):
                m = Ridge(alpha=a).fit(A, ytr[tr])
                cv[f, j] = rmse(ytr[va], m.predict(Av)); tr_err[f, j] = rmse(ytr[tr], m.predict(A)); nz[f, j] = A.shape[1]
    out = pd.DataFrame({"alpha": alphas, "cv": cv.mean(0), "se": cv.std(0, ddof=1) / np.sqrt(K),
                        "train": tr_err.mean(0), "nz": nz.mean(0)})
    i_min = out.cv.idxmin(); thr = out.cv[i_min] + out.se[i_min]
    a_1se = out.loc[out.cv <= thr, "alpha"].max()
    return out, out.alpha[i_min], a_1se, thr

t0 = time.time()
ridge_cv, ridge_min, ridge_1se, ridge_thr = cv_curve("ridge", alphas_ridge)
lasso_cv, lasso_min, lasso_1se, lasso_thr = cv_curve("lasso", alphas_lasso)
print(f"CV completada en {time.time()-t0:.1f} s")
for n, cvd, amin, a1 in [("Ridge", ridge_cv, ridge_min, ridge_1se), ("Lasso", lasso_cv, lasso_min, lasso_1se)]:
    r1 = cvd.set_index("alpha").loc[a1]; r0 = cvd.set_index("alpha").loc[amin]
    print(f"{n}: λ_min = {amin:.4g} (CV {r0.cv:.4f}) · λ_1SE = {a1:.4g} (CV {r1.cv:.4f}, coef. ≠ 0 ≈ {r1.nz:.0f} de {Xtr_np.shape[1]})")





fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (n, cvd, amin, a1, thr, col) in zip(axes, [("Ridge", ridge_cv, ridge_min, ridge_1se, ridge_thr, "#4C72B0"),
                                                  ("Lasso", lasso_cv, lasso_min, lasso_1se, lasso_thr, "#C44E52")]):
    ax.plot(cvd.alpha, cvd.train, "--", color="gray", label="Entrenamiento")
    ax.plot(cvd.alpha, cvd.cv, color=col, label="CV 10-fold")
    ax.fill_between(cvd.alpha, cvd.cv - cvd.se, cvd.cv + cvd.se, color=col, alpha=.2, label="± 1 SE")
    ax.axvline(amin, color="k", ls=":", lw=1, label=f"λ_min = {amin:.3g}")
    ax.axvline(a1, color=col, lw=1.4, label=f"λ_1SE = {a1:.3g}")
    ax.set_xscale("log"); ax.set(xlabel="λ (alpha, escala log)", ylabel="RMSE (log ingreso)", title=f"{n}: curva de validación")
    ax.legend(fontsize=8)
    if n == "Lasso":
        ax2 = ax.twiny(); ax2.set_xscale("log"); ax2.set_xlim(ax.get_xlim())
        ticks = cvd.alpha.values[::10]; ax2.set_xticks(ticks); ax2.set_xticklabels([f"{v:.0f}" for v in cvd.nz.values[::10]], fontsize=8)
        ax2.set_xlabel("n.º de coeficientes ≠ 0", fontsize=8); ax2.grid(False)
plt.tight_layout(); plt.savefig(FIG + "f3_curvas_lambda.png", dpi=150); plt.show() if "ipykernel" in sys.modules else plt.close()





sc_full = StandardScaler().fit(Xtr_np); A_s = sc_full.transform(Xtr_np)
al, coefs, _ = lasso_path(A_s, ytr - ytr.mean(), alphas=np.logspace(-4, 0, 80))
fig, ax = plt.subplots(figsize=(8, 4.2))
top = np.argsort(-np.abs(coefs[:, -1]))[:8]   # 8 coeficientes más grandes con λ mínimo
pal = plt.cm.tab10.colors
for j in range(coefs.shape[0]):
    if j in top: ax.plot(al, coefs[j], lw=1.8, color=pal[list(top).index(j)], label=cols[j])
    else: ax.plot(al, coefs[j], lw=.6, color="gray", alpha=.35)
ax.axvline(lasso_1se, color="#C44E52", lw=1.4, label="λ_1SE")
ax.set_xscale("log"); ax.set(xlabel="λ", ylabel="coeficiente estandarizado", title="Lasso: trayectoria de coeficientes")
ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(FIG + "f4_lasso_path.png", dpi=150); plt.show() if "ipykernel" in sys.modules else plt.close()


print("\n" + "=" * 70 + "\n3. Evaluación en el conjunto de prueba (15 pts)\n" + "=" * 70)




def final_models():
    sc = StandardScaler().fit(Xtr_np); A, At = sc.transform(Xtr_np), sc.transform(X_te.values)
    bs_idx = [cols.index(c) for c in BS_COLS]
    m = {}
    m["Modelo nulo (media)"] = (np.full(len(ytr), ytr.mean()), np.full(len(y_te), ytr.mean()), 0)
    ols = LinearRegression().fit(Xtr_np, ytr); m["OLS completo"] = (ols.predict(Xtr_np), ols.predict(X_te.values), Xtr_np.shape[1])
    bs = LinearRegression().fit(Xtr_np[:, bs_idx], ytr); m[f"Best subsets (k={k_1se})"] = (bs.predict(Xtr_np[:, bs_idx]), bs.predict(X_te.values[:, bs_idx]), len(bs_idx))
    for n, M, a in [("Ridge", Ridge, ridge_1se), ("Lasso", Lasso, lasso_1se)]:
        mod = M(alpha=a, max_iter=50000).fit(A, ytr)
        m[f"{n} (λ_1SE={a:.3g})"] = (mod.predict(A), mod.predict(At), int((mod.coef_ != 0).sum()))
    return m, ols

models, ols_full = final_models()
rows, preds_clp = [], {}
for n, (p_tr, p_te, npar) in models.items():
    s = np.mean(np.exp(ytr - p_tr))                 # factor smearing
    pc = np.exp(p_te) * s; preds_clp[n] = pc
    rows.append({"Modelo": n, "Predictores": npar, "Smearing": s,
                 "RMSE log": rmse(y_te, p_te), "MAE log": mean_absolute_error(y_te, p_te),
                 "R² log": 1 - np.sum((y_te - p_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2),
                 "MAE (CLP)": mean_absolute_error(ing_te, pc), "RMSE (CLP)": rmse(ing_te, pc),
                 "MedAE (CLP)": np.median(np.abs(ing_te - pc))})
res = pd.DataFrame(rows).set_index("Modelo")
display(res.style.format({"Smearing": "{:.3f}", "RMSE log": "{:.4f}", "MAE log": "{:.4f}", "R² log": "{:.3f}",
                          "MAE (CLP)": "${:,.0f}", "RMSE (CLP)": "${:,.0f}", "MedAE (CLP)": "${:,.0f}"}))
res.to_csv("resultados_test.csv")





B = 2000; n_te = len(ing_te); ref_err = np.abs(ing_te - preds_clp["OLS completo"])
idx_b = rng.integers(0, n_te, size=(B, n_te))
boot = []
for n, pc in preds_clp.items():
    if n == "OLS completo": continue
    diff = np.abs(ing_te - pc) - ref_err
    dist = diff[idx_b].mean(1)
    boot.append({"Modelo": n, "Δ MAE vs OLS (CLP)": diff.mean(),
                 "IC95 inf": np.percentile(dist, 2.5), "IC95 sup": np.percentile(dist, 97.5)})
boot = pd.DataFrame(boot).set_index("Modelo")
boot.to_csv("bootstrap_test.csv")
boot["¿Diferencia significativa?"] = np.where((boot["IC95 inf"] > 0) | (boot["IC95 sup"] < 0), "Sí", "No")
display(boot.style.format({c: "{:,.0f}" for c in ["Δ MAE vs OLS (CLP)", "IC95 inf", "IC95 sup"]}))





fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
best_name = [n for n in preds_clp if n.startswith("Lasso")][0]
pc = preds_clp[best_name]
ax[0].scatter(ing_te / 1e6, pc / 1e6, s=3, alpha=.25, color="#4C72B0")
ax[0].plot([0, 5], [0, 5], "k--", lw=1); ax[0].set(xlim=(0, 5), ylim=(0, 5), xlabel="Ingreso real (M CLP)", ylabel="Predicho (M CLP)",
                                                title=f"{best_name}: real vs predicho (prueba)")
dec = pd.qcut(ing_te, 10, labels=False)
err_dec = pd.DataFrame({"dec": dec + 1, "ae": np.abs(ing_te - pc)}).groupby("dec").ae.mean() / 1e3
ax[1].bar(err_dec.index, err_dec.values, color="#C44E52")
ax[1].set(xlabel="Decil de ingreso real", ylabel="MAE (miles CLP)", title="Error absoluto medio por decil de ingreso")
plt.tight_layout(); plt.savefig(FIG + "f5_test.png", dpi=150); plt.show() if "ipykernel" in sys.modules else plt.close()
print((err_dec.round(0)).to_dict())



print("\n" + "=" * 70 + "\n4. Curva de validación y compromiso sesgo-varianza (5 pts)\n" + "=" * 70)
# 

from sklearn.linear_model import RidgeCV
Xte_np = X_te.values
n_small, R = 150, 200
coef_ols, coef_ridge, err_ols, err_ridge, tr_ols, tr_ridge = [], [], [], [], [], []
for r in range(R):
    i = rng.choice(len(Xtr_np), n_small, replace=False)
    A, yA = Xtr_np[i], ytr[i]
    keep = A.std(0) > 0                                # columnas sin variación en la submuestra
    sc = StandardScaler().fit(A[:, keep]); As, Ts = sc.transform(A[:, keep]), sc.transform(Xte_np[:, keep])
    o = LinearRegression().fit(As, yA); rg = RidgeCV(alphas=np.logspace(-2, 4, 40), cv=5).fit(As, yA)
    err_ols.append(rmse(y_te, o.predict(Ts))); err_ridge.append(rmse(y_te, rg.predict(Ts)))
    tr_ols.append(rmse(yA, o.predict(As))); tr_ridge.append(rmse(yA, rg.predict(As)))
    full = lambda c: (lambda v: (v.__setitem__(keep, c), v)[1])(np.full(Xtr_np.shape[1], np.nan))
    coef_ols.append(full(o.coef_)); coef_ridge.append(full(rg.coef_))
var_ols = np.nanmean(np.nanvar(np.array(coef_ols), 0)); var_ridge = np.nanmean(np.nanvar(np.array(coef_ridge), 0))
exp_tab = pd.DataFrame({"RMSE entrenamiento": [np.mean(tr_ols), np.mean(tr_ridge)],
                        "RMSE prueba (media)": [np.mean(err_ols), np.mean(err_ridge)],
                        "RMSE prueba (desv.)": [np.std(err_ols), np.std(err_ridge)],
                        "Varianza media de coeficientes": [var_ols, var_ridge]}, index=["OLS", "Ridge (CV)"])
display(exp_tab)
print(f"Ridge reduce la varianza de los coeficientes en {100*(1-var_ridge/var_ols):.0f} % y el RMSE de prueba en {100*(1-np.mean(err_ridge)/np.mean(err_ols)):.0f} %.")





fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
ax[0].hist(err_ols, bins=30, alpha=.6, label="OLS", color="#C44E52"); ax[0].hist(err_ridge, bins=30, alpha=.6, label="Ridge", color="#4C72B0")
ax[0].set(xlabel="RMSE de prueba (log)", ylabel="n.º de réplicas", title=f"200 réplicas con n = {n_small}"); ax[0].legend()
j = cols.index("mujer")
ax[1].hist(np.array(coef_ols)[:, j], bins=30, alpha=.6, label="OLS", color="#C44E52")
ax[1].hist(np.array(coef_ridge)[:, j], bins=30, alpha=.6, label="Ridge", color="#4C72B0")
ax[1].set(xlabel="coeficiente estandarizado de 'mujer'", title="Dispersión de un coeficiente entre réplicas"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG + "f6_sesgo_varianza.png", dpi=150); plt.show() if "ipykernel" in sys.modules else plt.close()



print("\n" + "=" * 70 + "\n5. Conclusiones y recomendaciones de política pública (20 pts)\n" + "=" * 70)



co = pd.Series(ols_full.coef_, index=cols)
eff = (100 * (np.exp(co) - 1)).rename("efecto %")
sel = ["mujer", "informal", "extranjero", "rural"] + [c for c in cols if c.startswith(("educ_", "categoria_"))]
display(eff[sel].sort_values().to_frame().style.format("{:+.1f} %"))
reg = eff[[c for c in cols if c.startswith("region_")]].sort_values()
print("Regiones (vs Metropolitana):", reg.round(1).to_dict())
age_peak = -co["edad"] / (2 * co["edad2"]); print(f"Edad de ingreso máximo (ceteris paribus): {age_peak:.0f} años")
print(f"Elasticidad aprox. por hora adicional: {100*(np.exp(co['horas'])-1):.2f} % por hora")



import sklearn
print("Python", sys.version.split()[0], "· pandas", pd.__version__, "· numpy", np.__version__, "· scikit-learn", sklearn.__version__)

