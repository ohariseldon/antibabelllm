"""Treina a cabeça do detector sem chave (uma por idioma) e mede na régua.

  python treina.py estilo            só estilo
  python treina.py estilo emb lm     tudo junto
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from avalia import divisao, imprimir, relatorio
from config import resultado
from extrair import DADOS, carregar

BASE = DADOS.rsplit(".", 1)[0]


def matriz(blocos, linhas, lg):
    partes = []
    for b in blocos:
        if b == "estilo":
            partes.append(np.load(f"{BASE}.estilo_{lg}.npy"))
        elif b == "emb":
            partes.append(np.load(f"{BASE}.emb.npy"))
        elif b.startswith("lm"):
            X = np.load(f"{BASE}.lm_qwen0.5.npy")[:, 1:]
            partes.append(X)
    return np.concatenate(partes, 1)


def treinar(blocos, modelo="lr", salvar=None):
    linhas = carregar()
    split = divisao(linhas)
    y = np.array([l["rotulo"] for l in linhas])
    score = np.full(len(linhas), np.nan)
    modelos = {}
    for lg in ("pt", "en"):
        sel = np.array([l["lingua"] == lg for l in linhas])
        X = matriz(blocos, linhas, lg)
        ok = ~np.isnan(X[sel]).any(1)
        idx = np.where(sel)[0][ok]
        tr = idx[split[idx] == "treino"]
        if modelo == "lr":
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
        else:
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                 class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        score[idx] = clf.decision_function(X[idx]) if modelo == "lr" else clf.predict_proba(X[idx])[:, 1]
        modelos[lg] = clf
    ok = ~np.isnan(score)
    res = relatorio([l for l, k in zip(linhas, ok) if k], score[ok], split[ok],
                    titulo=f"{'+'.join(blocos)} ({modelo})")
    if salvar:
        json.dump(res, open(salvar, "w"), indent=1, ensure_ascii=False)
    return res, modelos, score


def _clf(modelo):
    if modelo == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          class_weight="balanced", random_state=0)


def _score(clf, X, modelo):
    return clf.decision_function(X) if modelo == "lr" else clf.predict_proba(X)[:, 1]


def cv_adversario(blocos, modelo="lr", k=5, com_babel=True):
    """pt: 5 dobras sobre as 20 famílias do babelllm. Em cada dobra, o treino vê as saídas
    do babelllm de 16 artigos e o teste é feito nos 4 artigos que ele nunca viu."""
    linhas = carregar()
    split = divisao(linhas)
    y = np.array([l["rotulo"] for l in linhas])
    sel = np.array([l["lingua"] == "pt" for l in linhas])
    X = matriz(blocos, linhas, "pt")
    ok = ~np.isnan(X).any(1) & sel
    fams = sorted({l["familia"] for l in linhas if l["fonte"].startswith("babel")})
    rng = np.random.RandomState(0)
    rng.shuffle(fams)
    dobras = [set(fams[i::k]) for i in range(k)]
    fam = np.array([l["familia"] for l in linhas])
    grupo = np.array([f"{l['fonte']}|{l['ataque']}" for l in linhas])
    acertos = {}
    fpr_real = []
    for d in dobras:
        no_teste = np.isin(fam, list(d))
        babel_fam = np.isin(fam, fams)
        tr = ok & (((split == "treino") & ~babel_fam) | (babel_fam & ~no_teste & com_babel))
        clf = _clf(modelo).fit(X[tr], y[tr])
        s = np.full(len(linhas), np.nan)
        s[ok] = _score(clf, X[ok], modelo)
        hc = s[ok & (split == "calib") & (y == 0)]
        ht = s[ok & (split == "teste") & (y == 0)]
        lim1, lim5 = np.quantile(hc, 0.99), np.quantile(hc, 0.95)
        fpr_real.append(((ht > lim1).mean(), (ht > lim5).mean()))
        for i in np.where(ok & no_teste & (y == 1))[0]:
            acertos.setdefault(grupo[i], []).append((s[i] > lim1, s[i] > lim5))
    print(f"== CV adversário pt ({'+'.join(blocos)}, {modelo}, treino {'com' if com_babel else 'sem'} babelllm)")
    print(f"   falso positivo real nos humanos de teste: 1%→{np.mean([a for a, _ in fpr_real]):.3f} "
          f"5%→{np.mean([b for _, b in fpr_real]):.3f}")
    tudo = []
    for g, v in sorted(acertos.items()):
        v = np.array(v)
        tudo.append(v)
        print(f"   {g:40s} n={len(v):4d} tpr@1%={v[:, 0].mean():.3f} tpr@5%={v[:, 1].mean():.3f}")
    v = np.concatenate(tudo)
    vb = np.concatenate([x for g, x in zip(sorted(acertos), tudo) if g.startswith("babel")])
    print(f"   {'SÓ BABELLLM':40s} n={len(vb):4d} tpr@1%={vb[:, 0].mean():.3f} tpr@5%={vb[:, 1].mean():.3f}")
    return acertos


def cv_modelo_novo(blocos, modelo="lr"):
    """en: deixa um gerador de fora do treino e mede nele (gerador que o detector nunca viu)."""
    linhas = carregar()
    split = divisao(linhas)
    y = np.array([l["rotulo"] for l in linhas])
    sel = np.array([l["lingua"] == "en" for l in linhas])
    X = matriz(blocos, linhas, "en")
    ok = ~np.isnan(X).any(1) & sel
    fonte = np.array([l["fonte"] for l in linhas])
    atk = np.array([l["ataque"] for l in linhas])
    print(f"== en, gerador nunca visto ({'+'.join(blocos)}, {modelo})")
    for m in sorted({f for f in fonte[ok] if f != "raid_human"}):
        tr = ok & (split == "treino") & (fonte != m)
        clf = _clf(modelo).fit(X[tr], y[tr])
        s = np.full(len(linhas), np.nan)
        s[ok] = _score(clf, X[ok], modelo)
        hc = s[ok & (split == "calib") & (y == 0)]
        lim1 = np.quantile(hc, 0.99)
        alvo = ok & (fonte == m) & (split != "treino")
        txt = " ".join(f"{a}={(s[alvo & (atk == a)] > lim1).mean():.2f}" for a in sorted(set(atk[alvo])))
        print(f"   {m:20s} tpr@1% total={(s[alvo] > lim1).mean():.3f}  {txt}")


if __name__ == "__main__":
    args = sys.argv[1:]
    modelo = "hgb" if "--hgb" in args else "lr"
    blocos = [a for a in args if not a.startswith("--")]
    if "--cv" in args:
        cv_adversario(blocos, modelo, com_babel=False)
        cv_adversario(blocos, modelo, com_babel=True)
        cv_modelo_novo(blocos, modelo)
    else:
        res, _, _ = treinar(blocos, modelo, salvar=resultado(f"res_{'_'.join(blocos)}_{modelo}.json"))
        imprimir(res)
