# Archive — Résultats SVD-latent et Tucker (post-compression latente)

> # ⚠️ LE CODE ET L'ARTICLE DIVERGENT VOLONTAIREMENT (2026-07-16)
>
> Le **code** a été changé (base SVD complexe + figée, cf. §5.2). L'**article**
> et **tous les chiffres de ce document** décrivent encore l'**ancien** code
> (base réelle, apprise, ajustée en domaine temporel). C'est **assumé** : on
> garde les anciens résultats tant qu'il n'y a pas de nouveaux runs.
>
> **À faire dès que les SVD sont re-lancées :**
> 1. Re-générer les 26 checkpoints SVD (§2) — leurs chiffres actuels ne
>    correspondent plus au code.
> 2. §2.3.2 de `main.tex` : `V ∈ R^{d_z × k_svd}` → complexe pour LLAE (réelle
>    pour SLAE, latents réels) ; préciser que `V` est **figée**.
> 3. §2.3.2 : supprimer la phrase « Because $V$ is real ... the two commute »
>    (fausse dès que `V` est complexe) ; la rétro-projection précède désormais
>    `L⁻¹`.
> 4. §2.3.2, paragraphe de perte : retirer « and, in the SVD variant, the
>    projection basis $V$ » de la liste des paramètres mis à jour.
> 5. §2.3.3 (Tucker) : supprimer « Unlike the real basis $V$ of the SVD
>    variant... » — l'opposition réel/complexe n'existe plus.
> 6. Envisager de **supprimer la SVD** : elle est désormais *exactement*
>    Tucker à `r_s = K` (vérifié numériquement, cf. §5.1).
>
> `tab:arch_summary` (Table 1) n'a **pas** besoin de changer : elle dessine déjà
> `MLP → V^⊤ → L⁻¹`, c'est-à-dire la rétro-projection avant l'inversion.

> Document d'archive. Contient l'intégralité des résultats des variantes de
> **post-compression latente** (SVD à un mode, Tucker à deux modes) telles
> qu'elles figuraient dans `main.tex` avant la condensation de ces sections.
> Rien n'est perdu ici : tout ce qui a été retiré ou déplacé en appendice dans
> l'article est conservé ci-dessous avec les chiffres exacts.
>
> **Convention de nommage (fixée le 2026-07-16) :** **POD nomme la méthode,
> SVD nomme l'opération.**
>
> - La **baseline linéaire** par fréquence (§2.4.1 de l'article, le point de
>   comparaison cité dans l'abstract à 8.91 %) s'appelle désormais **POD**
>   partout — c'est un pipeline ROM sur snapshots physiques, le POD-NN de
>   Hesthaven & Ubbiali. Son paramètre de modes est `k_pod`.
> - La **post-compression latente** décrite dans ce document reste **SVD** —
>   c'est bien une SVD tronquée des codes latents, et personne ne l'appellerait
>   POD. Son paramètre de modes est `k_svd`.
> - Les invocations purement algébriques (SVD randomisée de Halko, borne
>   d'Eckart–Young `E_SVD` de l'optimisation de contour) restent **SVD** : ce
>   sont des opérations, pas des méthodes.
>
> Ce document ne traite que la post-compression latente.

---

## 1. Méthodologie

### 1.1 Principe commun

Les deux variantes poursuivent le même but : réduire la dimension effective de
la cible du surrogate MLP avant de l'entraîner. Le surrogate direct prédit
`K · d_z` scalaires par échantillon (1024 pour `K=16`, `d_z=64`). SVD et Tucker
compressent cette cible hors ligne, puis rétro-projettent avant décodage.

Dans les deux cas la décomposition est appliquée **dans l'espace latent, après
la transformée de Laplace**.

### 1.2 SVD latente (un mode)

Approche inspirée des Rank Reduction Autoencoders (RRAE), `mounayer2025`.

Étant donnés les codes latents Laplace `{ẑ(s_k)}_{k=1..K}` collectés sur le
train, une SVD tronquée fournit une matrice de projection
`V ∈ R^{d_z × d_SVD}` (avec `d_SVD ≪ d_z`) telle que `ẑ(s_k) ≈ (ẑ(s_k) V) Vᵀ`.
Les codes réduits `Ĝ(s_k) = ẑ(s_k) V ∈ R^{d_SVD}` deviennent la cible du
surrogate — le problème passe de dimension `d_z` à `d_SVD`.

**Pipelines :**

- SLAE-SVD : `μ →[g_θ] G_pred(s_k) →[Vᵀ] z_pred(s_k) →[Decoder] Ũ(s_k) →[L⁻¹] U_rec(t)`
- LLAE-SVD : `μ →[g_θ] Ĝ_pred(s_k) →[Vᵀ] ẑ_pred(s_k) →[L⁻¹] z̃_pred(t) →[Decoder] U_rec(t)`

**Entraînement :** encodeur gelé ; décodeur **et** base `V` entraînés end-to-end
avec le surrogate. Perte totale :

```
L_total = α_lat · ‖w_k^pred − w_k^true‖² + α_spat · ‖U_rec − U‖²
```

La rétro-propagation depuis la MSE spatio-temporelle traverse le décodeur et la
transformée inverse, mettant à jour conjointement le surrogate, le décodeur, et
la base `V`.

### 1.3 Tucker (deux modes joints)

`kolda2009tensor`. Là où la SVD réduit un seul mode à la fois (pour LLAE-SVD :
collapse `d_z` ; pour SLAE-SVD : collapse `d_z` indépendamment à chaque
fréquence, **en ignorant les corrélations inter-fréquences**), Tucker factorise
simultanément deux modes du tenseur latent.

On assemble les latents fréquentiels du train en un tenseur d'ordre 3
`Ẑ ∈ C^{N_train × K × d_z}` (réel pour SLAE, complexe pour LLAE). Une
décomposition Tucker-`(r_μ, r_s, r_z)` approxime chaque élément :

```
Ẑ_{i,j,m} ≈ Σ_{p=1..r_μ} Σ_{l=1..r_s} Σ_{q=1..r_z}  Ĉ_{p,l,q} · U^{(μ)}_{i,p} · U^{(s)}_{j,l} · U^{(z)}_{m,q}
```

où `Ĉ ∈ C^{r_μ × r_s × r_z}` est le cœur global et `U^{(μ)}, U^{(s)}, U^{(z)}`
les matrices de facteurs orthogonaux pour les modes paramètre (échantillon),
fréquence et latent.

**Astuce clé pour la généralisation :** pour permettre la prédiction sur des
échantillons de test non vus, on absorbe les termes dépendant du paramètre dans
une matrice-cœur spécifique à la simulation `Ĝ_i ∈ C^{r_s × r_z}` :

```
(Ĝ_i)_{l,q} = Σ_{p=1..r_μ} Ĉ_{p,l,q} · U^{(μ)}_{i,p}
```

d'où la forme matricielle compacte pour la simulation `i` :

```
Ẑ_i ≈ U^{(s)} · Ĝ_i · U^{(z)ᵀ},     U^{(s)} ∈ C^{K × r_s},  U^{(z)} ∈ C^{d_z × r_z}
```

Les facteurs partagés `U^{(s)}` et `U^{(z)}` sont calculés **hors ligne** par
l'algorithme HOOI (Higher-Order Orthogonal Iteration), `delathauwer2000hooi`, et
restent **gelés**. Le surrogate apprend `μ ↦ Ĝ(μ) ∈ C^{r_s × r_z}`, de
dimension `r_s · r_z ≪ K · d_z`.

**Pipelines :**

- LLAE-Tucker : `μ →[g_θ] Ĝ_pred →[U^{(s)}, U^{(z)ᵀ}] ẑ_pred(s_k) →[L⁻¹] z̃_pred(t) →[Decoder] U_rec(t)`
- SLAE-Tucker : `μ →[g_θ] G_pred →[U^{(s)}, U^{(z)ᵀ}] z_pred(s_k) →[Decoder] Ũ(s_k) →[L⁻¹] U_rec(t)`

### 1.4 Positionnement vs POD-DL-ROM

Les étages SVD et Tucker utilisent des projections linéaires/multilinéaires,
mais dans un rôle **opposé** à POD-DL-ROM (`fresca2022`) : ils agissent *après*
l'autoencodeur non linéaire, dans l'espace latent, pour réduire la cible du
surrogate. C'est une **post-compression**, pas une pré-réduction spatiale.

---

## 2. Résultats — SVD latente

**Protocole :** 26 checkpoints AE+SVD sur les familles LLAE et SLAE.
`k_svd ∈ {16, 32}`. Encodeur gelé ; décodeur + base `V` entraînés end-to-end.
Erreur L² relative sur le champ physique `U(t)`, test set `n = 1620`.

### 2.1 Table complète des checkpoints

| Modèle | `d_z` | `K` | `γ` | `k_svd` | Médiane (%) | Moyenne (%) | p90 (%) |
|---|---|---|---|---|---|---|---|
| LLAE | 64  | 16 | 0.01  | 32 | **4.24** | 4.61 | 6.75 |
| LLAE | 64  | 32 | 0.01  | 32 | 4.29 | 4.69 | 7.02 |
| LLAE | 64  | 16 | 0.001 | 32 | 4.44 | 4.96 | 7.41 |
| LLAE | 64  | 16 | 0.0   | 32 | 4.50 | 4.97 | 7.29 |
| LLAE | 64  |  8 | 0.01  | 32 | 4.60 | 5.04 | 7.47 |
| LLAE | 128 | 16 | 0.01  | 32 | 4.77 | 5.30 | 7.82 |
| LLAE | 32  | 16 | 0.01  | 16 | 5.55 | 6.29 | 9.39 |
| LLAE | 64  | 32 | 0.01  | 16 | 5.58 | 6.22 | 9.35 |
| LLAE | 128 | 16 | 0.01  | 16 | 5.83 | 6.51 | 9.48 |
| LLAE | 64  | 16 | 0.01  | 16 | 5.84 | 6.58 | 9.79 |
| LLAE | 64  | 16 | 0.001 | 16 | 5.90 | 6.72 | 9.92 |
| LLAE | 64  | 16 | 0.0   | 16 | 5.94 | 6.76 | 10.18 |
| LLAE | 64  |  8 | 0.01  | 16 | 6.54 | 7.37 | 11.05 |
| SLAE | 64  | 32 | 0.01  | 32 | **6.03** | 6.58 | 9.02 |
| SLAE | 64  | 32 | 0.01  | 16 | 6.96 | 7.80 | 11.24 |
| SLAE | 64  | 16 | 0.0   | 32 | 7.50 | 7.73 | 9.26 |
| SLAE | 64  | 16 | 0.001 | 32 | 7.62 | 7.88 | 9.52 |
| SLAE | 64  | 16 | 0.01  | 32 | 7.90 | 8.32 | 10.54 |
| SLAE | 32  | 16 | 0.01  | 16 | 7.95 | 8.38 | 10.71 |
| SLAE | 64  | 16 | 0.001 | 16 | 8.01 | 8.43 | 10.80 |
| SLAE | 128 | 16 | 0.01  | 32 | 8.12 | 8.57 | 10.98 |
| SLAE | 64  | 16 | 0.0   | 16 | 8.13 | 8.61 | 11.06 |
| SLAE | 64  | 16 | 0.01  | 16 | 8.71 | 9.40 | 12.68 |
| SLAE | 128 | 16 | 0.01  | 16 | 8.89 | 9.71 | 13.43 |
| SLAE | 64  |  8 | 0.01  | 32 | 11.64 | 11.93 | 13.96 |
| SLAE | 64  |  8 | 0.01  | 16 | 11.97 | 12.38 | 14.88 |

Plages globales : **LLAE 4.2–6.5 %**, **SLAE 6.0–12.0 %**.

### 2.2 Comprimer le latent aide-t-il ? — Non

Question centrale : prédire sur un espace latent plus petit et orthogonalisé
simplifie-t-il assez la tâche du MLP pour compenser l'information perdue par la
troncature SVD ? **Non.**

À **configuration d'autoencodeur identique** (`d_z=64`, `γ=0.01`), le
goulot SVD ajoute une petite pénalité vs le surrogate direct, pour les deux
familles :

| Famille | Meilleur SVD | Direct correspondant | Écart |
|---|---|---|---|
| LLAE | 4.24 % (`K=16`, `k_svd=32`) | 4.18 % | **+0.06 pp** |
| SLAE | 6.03 % (`K=32`, `k_svd=32`) | 5.69 % | **+0.34 pp** |

**Interprétation :** le MLP navigue très bien l'espace latent complet de
dimension `d_z`. La projection SVD ne fait que jeter du détail spatial que le
surrogate direct avait déjà appris à prédire. Comprimer le latent *avant* le
surrogate dégrade la reconstruction au lieu de l'aider.

### 2.3 Impact de `k_svd` (16 vs 32)

Comparaison de paires de configurations ne différant que par `k_svd`.
`k_svd = 32` améliore **systématiquement** la précision sur tous les modèles
(tous les points sous la diagonale `y = x`).

| Famille | Réduction d'erreur médiane (16 → 32) |
|---|---|
| LLAE | **+1.06 à +1.94 pp** |
| SLAE | **+0.33 à +0.93 pp** |

**Interprétation :** les espaces latents LLAE contiennent davantage
d'information fine utile, captée par les 16 modes SVD supplémentaires. SLAE
semble déjà jeter du détail spatial au stade de l'AE, ce qui limite le gain
potentiel d'un `k_svd` plus grand.

### 2.4 Sensibilité aux hyperparamètres AE

La sensibilité à `K`, `d_z`, `γ` reflète le cas direct : SLAE se dégrade
brutalement à `K=8` (**11.6–12.0 %**), tandis que LLAE reste dans une bande
étroite. Le seul levier spécifique à ce pipeline est `k_svd` (§2.3).

### 2.5 Latence d'inférence

NVIDIA RTX A5000, batch size 1 :

| Modèle | Latence |
|---|---|
| LLAE-SVD | ≈ 22 ms/échantillon |
| SLAE-SVD | ≈ 9 ms/échantillon |

SLAE-SVD conserve son avantage de vitesse marqué (**≈ 2.4×**) sur LLAE-SVD,
reflet de son décodeur plus léger. L'étape de décodage SVD (multiplication
matricielle) entre la prédiction MLP et le décodeur spatial ajoute un léger
surcoût — la variante SVD est marginalement la plus lente des trois stratégies,
à cause de sa rétro-projection par fréquence.

### 2.6 Figures associées

- `svd_surr_medians.png` — comparaison par checkpoint : erreur médiane L² relative
  avec barres d'erreur IQR, groupées par famille.
- `svd_surr_ksvd_impact.png` — gauche : scatter des médianes appariées montrant
  que `k_svd=32` bat systématiquement `k_svd=16`. Droite : réduction absolue
  d'erreur en passant de 16 à 32 modes, triée par magnitude.
- `svd_surr_latency.png` — gauche : latence par échantillon sur les checkpoints.
  Droite : scatter précision–latence.

---

## 3. Résultats — Tucker

**Protocole :** grille de rangs `r_s ∈ {4, 8}` (mode fréquence, `K = 16 → r_s`)
× `r_z ∈ {8, 16, 32}` (mode latent, `d_z = 64 → r_z`), pour les deux familles →
**12 checkpoints**. Tous partagent le **même autoencodeur de Phase 1**
(`d_z = 64`, `K = 16`, `γ = 0.01`) ; seuls les facteurs hors ligne et le
surrogate diffèrent.

Ratio de compression : `ρ = r_s·r_z / (K·d_z)`, mesurant la réduction de la
dimension-cible du surrogate vs le pipeline direct (qui prédit `K·d_z = 1024`
scalaires/échantillon).

### 3.1 Les deux diagnostics sans décodeur

Au-delà de l'erreur de champ physique `ε_field` (L² relatif sur `U(t)`,
pipeline complet), deux diagnostics calculés **dans l'espace latent** séparent
la difficulté entre la compression hors ligne et le MLP :

```
ε_trunc = ‖Ẑ − Π Ẑ‖ / ‖Ẑ‖              (facteurs seuls, indépendant du surrogate)
ε_reg   = ‖Ĝ_pred − Ĝ_true‖ / ‖Ĝ_true‖  (surrogate seul, indépendant de la troncature)
```

où `Π = U^{(s)} U^{(s)ᵀ}(·) U^{(z)} U^{(z)ᵀ}` est le projecteur Tucker.

**Pourquoi dans l'espace latent et pas physique :** voir §3.6 (co-adaptation du
décodeur) — un oracle en espace physique n'est pas une borne inférieure valide
pour SLAE-Tucker.

### 3.2 Table complète des 12 checkpoints

Test set `n = 1620`, trié par médiane dans chaque famille. `Lat` = latence GPU
batch size 1 (RTX A5000).

| Modèle | `r_s` | `r_z` | `ρ` | Médiane (%) | Moyenne (%) | p90 (%) | `ε_trunc` (%) | `ε_reg` (%) | Lat (ms) |
|---|---|---|---|---|---|---|---|---|---|
| LLAE | 8 | 32 | 0.250 | **4.33** | 4.75 | 7.16 | 36.98 | 7.25 | 18.2 |
| LLAE | 4 | 32 | 0.125 | 4.57 | 5.04 | 7.61 | 40.34 | 5.67 | 18.0 |
| LLAE | 8 | 16 | 0.125 | 4.87 | 5.43 | 8.11 | 61.86 | 4.38 | 18.3 |
| LLAE | 4 | 16 | 0.062 | 5.16 | 5.77 | 8.70 | 61.59 | 3.83 | 18.0 |
| LLAE | 8 |  8 | 0.062 | 6.85 | 7.78 | 11.75 | 81.21 | 4.99 | 18.3 |
| LLAE | 4 |  8 | 0.031 | 6.87 | 8.02 | 12.73 | 81.03 | 3.80 | 18.0 |
| SLAE | 8 | 32 | 0.250 | **7.92** | 8.35 | 10.55 | 42.40 | 6.76 | 7.4 |
| SLAE | 4 | 32 | 0.125 | 8.15 | 8.68 | 11.14 | 50.82 | 6.58 | 7.1 |
| SLAE | 4 | 16 | 0.062 | 8.68 | 9.36 | 12.79 | 69.59 | 5.18 | 7.1 |
| SLAE | 8 | 16 | 0.125 | 8.85 | 9.51 | 12.89 | 67.59 | 5.83 | 7.5 |
| SLAE | 4 |  8 | 0.031 | 9.93 | 10.92 | 15.61 | 80.63 | 5.70 | 7.1 |
| SLAE | 8 |  8 | 0.062 | 9.96 | 11.05 | 16.00 | 80.05 | 7.16 | 7.5 |

### 3.3 Précision globale

À AE partagé (`d_z=64`, `K=16`, `γ=0.01`), l'erreur médiane croît **doucement**
le long de l'axe de compression :

| Famille | Direct | → SVD (`k_svd=32`) | → Tucker (`r_s=8`, `r_z=32`) |
|---|---|---|---|
| LLAE | 4.18 % | 4.24 % | 4.33 % |
| SLAE | 7.59 % | 7.90 % | 7.92 % |

Tucker ajoute donc **au plus +0.15 pp** sur le surrogate direct et **+0.09 pp**
sur SVD — tout en réduisant la cible à `r_s·r_z = 256` scalaires :
**4× moins** que la cible directe (`K·d_z = 1024`) et **2× moins** que la cible
SVD (`K·k_svd = 512`).

**C'est la factorisation jointe des deux modes qui achète ça :** exploiter les
corrélations inter-fréquences permet à Tucker d'égaler la précision de la SVD
à un mode avec une cible de dimension moitié moindre.

### 3.4 ⭐ Sensibilité aux rangs : `r_z` domine `r_s`

**Résultat le plus important de la section.** La grille est fortement
**anisotrope** :

| Variation | Effet sur l'erreur LLAE |
|---|---|
| `r_z` : 8 → 32 (à `r_s=8`) | 6.85 % → 4.33 % — **quasi divisée par 2** |
| `r_s` : 4 → 8 (à `r_z=32`) | 4.57 % → 4.33 % — **bouge à peine** |

Les sweeps confirment le motif pour les deux familles : les violons `r_z` se
séparent nettement, les violons `r_s` se chevauchent.

**Interprétation (à relier à la thèse centrale du papier) :** les `K=16`
fréquences de Laplace sont **déjà fortement corrélées** entre elles sur le
training set — une base fréquentielle de rang 4 capture presque toute leur
variation conjointe. À l'inverse, les `d_z=64` canaux latents portent une
information plus riche et moins redondante, qu'une base de rang 8 tronque trop
agressivement.

C'est une **confirmation indépendante** que la représentation de Laplace est
intrinsèquement compacte — obtenue par un chemin (factorisation tensorielle)
totalement distinct du reste du papier.

**Conséquence pratique :** `r_s = 4` est le choix efficace en paramètres — il
**divise par 2 la taille du surrogate** (1.35 M vs 1.9 M paramètres pour
`r_s=8`) à coût de précision négligeable.

### 3.5 Troncature vs régression : où est le goulot

Les deux diagnostics localisent le goulot de précision.

**`ε_reg` reste petit partout (3.8–7.3 %)** : le MLP prédit le cœur compact de
façon fiable. Il croît seulement légèrement quand le cœur grossit (`r_z=32` est
plus dur à ajuster que `r_z=8`).

**`ε_trunc` est grand et c'est lui que l'erreur de champ suit** : il tombe de
~81 % à `r_z=8` à ~37 % à `r_z=32` (LLAE), reflétant la chute de l'erreur de
champ.

> **Nuance importante à ne pas perdre :** `ε_trunc` est une **énergie en espace
> latent**, pas une erreur de champ. Jeter 80 % de la norme latente ne coûte que
> ~7 % sur `U(t)`, parce que les directions Tucker supprimées portent peu
> d'information décodable. C'est sa **variation** avec les rangs, pas sa valeur
> absolue, qui gouverne la précision.

**Conclusion :** c'est la **compression Tucker**, pas le surrogate, qui est la
contrainte limitante. Le MLP n'est jamais le facteur limitant — donc tout gain
de précision doit venir d'une base latente plus riche (`r_z` plus grand), pas
d'un prédicteur plus gros.

### 3.6 ⭐ Co-adaptation du décodeur (avertissement méthodologique)

Conséquence subtile du fine-tuning end-to-end. Comme le décodeur est mis à jour
conjointement pendant l'entraînement du surrogate (`lr_decoder = 5e-5`), il peut
**co-adapter à la distribution des cœurs *prédits*** plutôt qu'aux vrais.

**Protocole de sonde :** décoder le cœur projeté exact à la place de la
prédiction du surrogate. Si le décodeur n'avait pas co-adapté, le cœur exact
donnerait une erreur plus faible et servirait d'oracle (borne inférieure valide).

| Famille | Effet de substituer le cœur exact | Conclusion |
|---|---|---|
| LLAE-Tucker | **+0.07 pp** (neutre) | la borne oracle **tient** |
| SLAE-Tucker | **+11.1 pp** (sonde à `r_s=8`, `r_z=16`) | la borne oracle **ne tient pas** |

Pour SLAE-Tucker, décoder le cœur exact **augmente fortement** l'erreur : le
décodeur SLAE s'est **spécialisé à la distribution des cœurs prédits**. C'est
exactement le mécanisme qui produit le **Gap surrogate négatif** observé pour
SLAE en §direct_surr.

**⇒ Un oracle en espace physique n'est donc pas une borne inférieure valide pour
SLAE-Tucker.** C'est la raison pour laquelle les diagnostics de troncature et de
régression (§3.1) sont calculés dans l'espace latent, où ils restent
indépendants du décodeur.

### 3.7 Compromis de compression (Pareto)

Le long du front de Pareto, LLAE-Tucker se dégrade **gracieusement** à mesure
que la cible est comprimée :

| Config | `ρ` | Médiane | Réduction de cible vs direct |
|---|---|---|---|
| `r_s=4`, `r_z=32` | 0.125 | 4.57 % | **8×** |
| `r_s=4`, `r_z=16` | 0.062 | 5.16 % | **16×** |

Tucker offre donc un **cadran précision/compacité réglable** que les surrogates
direct et SVD n'ont pas : quand la sortie du surrogate doit rester petite —
stockage, prédicteur léger, usage many-query en aval — Tucker soutient une
précision quasi-directe à une fraction de la dimension-cible.

Il ne **bat** cependant pas le surrogate direct en précision brute : comme pour
la SVD, réduire l'espace latent avant le surrogate jette de l'information que le
prédicteur direct est capable d'apprendre.

### 3.8 Latence d'inférence

La rétro-projection Tucker (`U^{(s)} Ĝ U^{(z)ᵀ}`, deux petits produits
matriciels) est **négligeable** dans le budget d'inférence.

RTX A5000, batch size 1, 50 runs chronométrés par CUDA Events :

| Modèle | Latence |
|---|---|
| LLAE-Tucker | ~18 ms/échantillon |
| SLAE-Tucker | ~7 ms/échantillon |

Essentiellement **indépendante des rangs** et virtuellement identique aux
pipelines direct et SVD. SLAE-Tucker préserve son avantage de latence
**≈ 2.4×** (décodeur plus léger), au prix du plancher d'erreur plus élevé.

### 3.9 Figures associées

- `tucker_grid.png` — erreur médiane de champ sur la grille `(r_s, r_z)`, LLAE
  (gauche) et SLAE (droite), même échelle de couleur. **Montre l'anisotropie.**
- `tucker_rank_sweep.png` — sweeps de rangs (violons, poolés sur l'autre rang ;
  ligne noire = médiane, orange pointillé = moyenne). Haut : LLAE-Tucker, bas :
  SLAE-Tucker ; gauche : effet `r_s`, droite : effet `r_z`.
- `tucker_error_decomp.png` — décomposition d'erreur LLAE (gauche) / SLAE
  (droite) : `ε_trunc`, `ε_reg`, `ε_field` pour chaque paire `(r_s, r_z)`
  (échelle log). L'erreur de champ suit la troncature, pas la régression.
- `tucker_pareto.png` — gauche : erreur médiane vs ratio `ρ` (pointillés = front
  de Pareto par famille ; les points partageant un ratio diffèrent en
  `(r_s, r_z)`). Droite : erreur médiane vs paramètres entraînables du surrogate.
- `tucker_bench.png` — benchmark d'inférence. Gauche : latence par checkpoint.
  Droite : scatter précision–latence. La latence est fixée par la famille de
  décodeur, pas par les rangs Tucker.
- `tucker_histograms.png` — histogrammes d'erreur par échantillon pour les 12
  checkpoints (SLAE en bleu, LLAE en rouge ; pointillé noir = médiane, orange =
  moyenne, rouge pointillé = p90).

---

## 4. Verdict — La compression latente en vaut-elle la peine ?

Trois stratégies de surrogate — **direct**, **SVD à un mode**, **Tucker à deux
modes joints** — ne différant que par la façon dont la cible latente est
comprimée.

### 4.1 Sur le coût d'inférence : non, et la raison est structurelle

Les trois pipelines **partagent le même décodeur**, et c'est le décodeur — qui
reconstruit les `K` frames de Laplace (SLAE) ou les `N_t` frames temporelles
(LLAE) — qui **domine l'inférence**.

SVD et Tucker ne changent que la largeur de la tête de sortie du MLP et
insèrent une petite rétro-projection linéaire (`Vᵀ`, ou les deux facteurs
Tucker) avant décodage. **Ni l'un ni l'autre ne touche la charge du décodeur.**

La latence est donc **effectivement invariante** à la stratégie de compression —
dans chaque famille les trois pipelines tombent dans la même bande :
**SLAE ~7–9 ms**, **LLAE ~18–22 ms** (batch 1, RTX A5000). La variante SVD est
marginalement la plus lente, à cause de sa rétro-projection par fréquence.

### 4.2 Table de synthèse (AE partagé : `d_z=64`, `K=16`, `γ=0.01`)

| Famille | Stratégie | Cible (coeff.) | Médiane (%) | Latence (ms) |
|---|---|---|---|---|
| LLAE | Direct | 1024 | 4.18 | 18.3 |
| LLAE | SVD (`k_svd=32`) | 512 | 4.24 | ~22 |
| LLAE | Tucker (`r_s=8`, `r_z=32`) | **256** | 4.33 | 18.2 |
| SLAE | Direct | 1024 | 7.59 | 8.4 |
| SLAE | SVD (`k_svd=32`) | 512 | 7.90 | ~9 |
| SLAE | Tucker (`r_s=8`, `r_z=32`) | **256** | 7.92 | 7.4 |

*Cible (coeff.)* = nombre de coefficients latents que le surrogate prédit par
échantillon (`K·d_z`, `K·k_svd`, `r_s·r_z` respectivement ; complexes pour LLAE,
réels pour SLAE). Test set `n=1620`.

**Lecture :** la compression réduit la cible jusqu'à **4×** mais laisse la
précision et la latence **essentiellement inchangées**.

### 4.3 Le verdict

Relativement au surrogate direct, ni SVD ni Tucker n'améliore la **précision**
(chacun ajoute une fraction de point) ni la **latence** (le pipeline est
decoder-bound).

Ce qu'ils **achètent** : une cible de prédiction plus petite et plus simple —
Tucker jusqu'à **256 coefficients**, réduction **4×** sur le direct
(`K·d_z = 1024`).

Cette compacité ne paie que quand le **surrogate**, et non le décodeur, est la
contrainte limitante :
- budget de données d'entraînement serré,
- stockage ou transmission des coefficients prédits,
- analyse en aval du cœur Tucker compact.

**Pour le problème présent, où le décodeur gouverne à la fois le coût et
l'erreur, le surrogate _direct_ est le choix pragmatique.** La valeur de SVD et
Tucker est **principalement méthodologique** : elle montre que l'essentiel du
contenu prédictif du latent survit à une réduction de rang agressive.

---

## 5. État de l'article (2026-07-16) et décisions en suspens

**Structure retenue : SVD et Tucker restent DEUX sections séparées**, en méthodo
(`sec:svd_compression`, `sec:tucker_compression`) comme en résultats
(`sec:ae_svd_surr`, `sec:tucker_surr`). Une fusion en une sous-section unique
avait été essayée puis annulée, le temps de trancher les points ci-dessous.

**Renforcé :** le paragraphe d'anisotropie de §Tucker insiste désormais sur le
fait qu'une factorisation tensorielle, ajustée à l'aveugle sur les statistiques
d'entraînement, redécouvre la prémisse du papier (la représentation de Laplace
est intrinsèquement compacte selon l'axe fréquence). Une réserve honnête y est
posée : l'axe `r_s` n'a que **deux points** ({4, 8}), donc on établit
l'insensibilité sur cette plage, pas la saturation.

### 5.1 SVD ≡ Tucker à `r_s = K` — établi

À `r_s = K`, `U_s` est carrée orthonormée donc unitaire :
`U_s G U_zᴴ = U_s U_sᴴ Ẑ U_z U_zᴴ = Ẑ U_z U_zᴴ`. Le mode fréquence n'est pas
compressé, la cible fait `K·r_z` = celle de la SVD avec `V = U_z`,
`k_svd = r_z`. **Architecturalement, SVD est Tucker à `r_s = K`** — la Table 1
(`tab:arch_summary`) dessine d'ailleurs déjà les deux lignes identiques au label
de boîte près. `hooi()` fait `r_s = min(r_s, K)`, donc `r_s=16` tourne sans
modification de code.

Ce qui **reste** distinct après le correctif du §5.2 : `V` est **réelle et
apprise** (`lr_V=5e-4`), `U_z` est **complexe et gelée**. Une `V` réelle commute
avec `L⁻¹` (l'inverse de Tikhonov est ℝ-linéaire : il prend `Re()` et étend par
conjugaison) ; un `U_z` complexe **ne commute pas**, d'où l'ordre de
rétro-projection différent dans les deux pipelines.

**Pour SLAE il n'y a aucun écart** : `svd_lowrank(Z_flat)` sur le dépliage
`(N·K, D)` donne littéralement l'initialisation HOSVD de `U_z`. Les chiffres le
confirment (`r_z=32` : 8.15 → 7.92 → 7.90 pour `r_s` = 4, 8, SVD). **SLAE est le
témoin qui valide l'identité.**

### 5.2 Changement de code appliqué (2026-07-16) — option (A)

Trois positions cohérentes étaient possibles pour LLAE-SVD :

- **(A)** `V` complexe et figée → la SVD devient *exactement* Tucker à `r_s = K`.
- **(B)** `V` réelle ajustée en Laplace → colle au texte littéral de §2.3.2, mais
  asymétrique avec Tucker (complexe) sans justification défendable.
- **(C)** `V` réelle ajustée sur `z(t)` → l'ancien code, auto-cohérent
  (« comprimer la trajectoire réelle, puis transformer »).

**(A) a été retenue.** Motif décisif : `ẑ V Vᵀ` avec `V` réelle projette sur un
sous-espace complexe qui *admet une base réelle* — un sous-ensemble **strict** de
ce qu'un `U_z` complexe atteint. À rang égal et cible de taille égale, le
complexe est donc **strictement plus expressif**, et la contrainte réelle
n'achetait que la commutation (un détail d'efficacité).

Changements :

| Fichier | Changement |
|---|---|
| `models/llae_svd_surrogate.py` | `V` : `nn.Parameter` réelle → `register_buffer` **complexe figée** ; rétro-projection `@ V^H` **avant** `L⁻¹` ; cible `Ĝ = ẑ V` |
| `models/slae_svd_surrogate.py` | `V` : `nn.Parameter` → `register_buffer` **figée** (reste réelle, latents SLAE réels) |
| `lightning/llae_svd_surrogate_module.py` | fit = `_truncated_left` (de `tucker.py`) sur le dépliage mode-latent de `Ẑ` ; groupe `lr_V` retiré |
| `lightning/slae_svd_surrogate_module.py` | groupe `lr_V` retiré |
| `scripts/train_surrogate.py` | `use_svd` détecte via `k_svd` seul (`lr_V` n'existe plus) |
| `configs/training/surrogate_{llae,slae}_svd.yaml` | `lr_V` retiré, commentaires mis à jour |

**Vérifié numériquement** (`scratchpad/test_equiv.py`) : `V` complexe
orthonormée ; `U_s` unitaire à `r_s=K` ; projecteurs latents identiques ;
`max |rec_svd − rec_tucker| = 1.3e-06`. **L'équivalence tient dans le code.**

⚠️ Non testé de bout en bout (pas de dataset ni de checkpoints en local) — à
valider au premier run distant. Points de vigilance : `torch.linalg.svd` complexe
sur le dépliage `(64, ns·K)`, et le chargement des anciens checkpoints SVD qui
échouera (`V` a changé de dtype et de statut Parameter→buffer).

### 5.3 Anomalie ouverte

En posant la SVD comme colonne `r_s=16` de la grille LLAE :

| | `r_s=4` | `r_s=8` | `r_s=16` (= SVD) |
|---|---|---|---|
| `r_z=32` | 4.57 | 4.33 | **4.24** ✓ cohérent |
| `r_z=16` | 5.16 | 4.87 | **5.84** ✗ +0.97 pp dans le mauvais sens |

Cesser de compresser le mode fréquence ne peut pas dégrader l'erreur, et une `V`
*apprise* devrait faire mieux qu'un `U_z` gelé. Hypothèse : l'ajustement en
domaine temporel (§5.2), qui diverge du sous-espace de Laplace à bas rang et se
recouvre à rang 32. **Non prouvé** — le correctif du §5.2 devrait faire tomber
le 5.84 s'il est la cause.

### 5.4 Décisions en suspens

1. **Supprimer la SVD, ou re-runner ?** Coût nul pour supprimer (retire 2 modèles,
   2 modules Lightning, 2 configs, 2 lignes de Table 1, une table, 3 figures ;
   neuf pipelines → sept). Sinon 6 runs de phase 2 (`r_s ∈ {4,8,16}` ×
   `r_z ∈ {8,16,32}` × 2 familles) pour compléter la grille en 3×3 et solidifier
   l'anisotropie sur trois points.
2. **Re-runner les 26 checkpoints SVD** après le correctif §5.2 (obligatoire si
   la SVD reste).
3. Comparaison à **cible égale** disponible sans aucun run : à 256 coefficients,
   Tucker 4.33 vs SVD 5.84 (LLAE) et 7.92 vs 8.71 (SLAE) ; Tucker à 128
   (`r_s=8, r_z=16`, 4.87) bat encore la SVD à 256.
