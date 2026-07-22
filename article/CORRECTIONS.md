# Revue de l'article vs code — corrections à apporter

Vérification systématique de `article/main.tex` contre le code du dépôt (transformées,
modèles, modules Lightning, configs, scripts). Classement par gravité.
Références : lignes de `main.tex` et fichiers sources.

---

## 1. Problèmes méthodologiques majeurs

### 1.1 Le dataset est une augmentation par rotation, pas 8 100 simulations indépendantes
**Où :** §3.1 (l. 612–619) — « The system is governed by … θ: Injection angle » ; « The dataset comprises 6480 training simulations and 1620 test simulations ».

**Code :** `src/laplace_surrogate/utils/rotate.py`, `dataset/doe_rotated.npy`.
Le DOE réel est une grille 15 (k) × 15 (C) = **225 simulations CFD**, chacune dupliquée en
**36 copies tournées numériquement** (`cv2.warpAffine`, angles 0–350°, rotation à 200×200 puis
center-crop 128×128). Le paramètre « injection angle » est donc une rotation d'image, pas un
paramètre simulé indépendamment.

**Conséquence :** le split train/test est aléatoire sur les 8 100 échantillons augmentés
(`make_split.py`), donc pour chaque cas test (k, θ, C), le train contient très probablement la
**même simulation de base (k, C) sous d'autres angles**. Les erreurs test sont optimistes : la
dimension « angle » est résoluble par pure équivariance de rotation. C'est le point le plus
important à corriger pour éviter une erreur méthodologique :
- au minimum, divulguer explicitement la construction (225 sims × 36 rotations, crop 200→128) ;
- idéalement, refaire (ou ajouter) un split par simulation de base (grouper les 36 rotations
  d'un même (k, C) dans le même fold) et reporter cette erreur-là comme mesure de généralisation.

À noter aussi : « The spatial domain is discretized into a uniform grid of 128×128 » — en réalité
c'est un **center-crop** du domaine natif 200×200 ; le domaine Ω décrit n'est pas le domaine simulé complet.

### 1.2 LLAE-SVD : la SVD est ajustée sur les latents temporels, pas sur les codes Laplace
**Où :** §2.4.2 (l. 255–266) — « For both SLAE and LLAE, the SVD is applied in the frequency
domain: given the K Laplace-domain latent codes {ẑ(s_k)} … a truncated SVD yields V » et
« In the LLAE-SVD variant, the SVD is likewise applied in the Laplace domain ».

**Code :** `llae_svd_surrogate_module.py` (`_compute_svd_and_stats`) et
`llae_svd_surrogate.py` (docstring) : la SVD tronquée est calculée sur
**z(t) dans le domaine temporel** (`z_all[train].reshape(-1, D)` réel), la base V ∈ R^{d_z×k_svd}
est appliquée **avant** la transformée de Laplace : G(t) = z(t)V, puis Ĝ(s_k) = L{G(t)}.
Le docstring de `llae_tucker_surrogate.py` le confirme : « Contrairement à SVD (une seule base
latente V appliquée AVANT Laplace)… ».

Détails :
- Les **cibles** du surrogate coïncident mathématiquement (L{z(t)V} = ẑ(s_k)V, opérations
  linéaires sur des axes différents), mais la **base V n'est pas celle qui minimise le résidu
  dans le domaine de Laplace** — l'affirmation « SVD applied in the frequency domain » est fausse
  pour LLAE.
- Incohérence interne de l'article : une SVD tronquée de codes complexes ẑ(s_k) donnerait une
  base **complexe**, or l'article écrit V ∈ R^{d_z×d_SVD}. Le V réel n'est cohérent qu'avec un
  fit sur z(t) réel.
- Le pipeline LLAE-SVD (l. 264–266 et tableau récapitulatif l. 570–582) montre l'ordre
  MLP → Vᵀ → L⁻¹ ; le code fait MLP → L⁻¹ → Vᵀ (`_predict_and_reconstruct`). Équivalent
  mathématiquement, mais autant écrire l'ordre réellement implémenté (ou signaler l'équivalence).

**Correction suggérée :** décrire LLAE-SVD comme dans le code (SVD sur la séquence latente
temporelle, transformée de Laplace sur les coefficients réduits — c'est d'ailleurs le schéma
LSLAE de CLAUDE.md), et réserver « SVD dans le domaine de Laplace » au seul SLAE-SVD
(base unique V fittée sur les latents z_k **poolés sur toutes les fréquences**, cf. §1.3).

### 1.3 §3.6 : « the spatial AE is kept frozen » est faux
**Où :** §3.6 (l. 921) — « In these models, the spatial AE is kept frozen ».

**Code :** `llae_svd_surrogate_module.configure_optimizers` (et SLAE équivalent) : le
**décodeur est fine-tuné** (`lr_decoder = 5e-5`, groupe de paramètres inconditionnel) et la
**base V est entraînée** (`lr_V = 5e-4`). Seul l'encodeur est gelé. C'est aussi en contradiction
avec le §2.4.2 de l'article lui-même (l. 272 : « jointly updating the surrogate weights, the AE
decoder, and, in the SVD variant, the projection basis V »). Corriger la phrase du §3.6
(« the encoder is kept frozen; the decoder and V are fine-tuned »).

### 1.4 Tucker : la décomposition implémentée est un Tucker-2 (HOOI sur 2 modes), et il manque les conjugués
**Où :** §2.4.3 (l. 278–288).

**Code :** `models/tucker.py` : « Le mode échantillon (0) n'est pas factorisé ». Il n'existe
**aucun facteur U^(μ)** ni rang r_μ : HOOI ne factorise que les modes fréquence et latent, et la
core par simulation est obtenue par **projection directe** G_i = U_sᴴ Ẑ_i U_z.

Problèmes dans le texte :
1. « A Tucker-(r_μ, r_s, r_z) decomposition… » avec le facteur U^(μ) et l'astuce d'absorption
   (l. 280–283) décrit un algorithme qui n'est pas celui du code. Si r_μ < N_train était
   réellement utilisé, les cibles d'entraînement seraient différentes. Réécrire : décomposition
   de Tucker **partielle (Tucker-2)** sur les modes (fréquence, latent), mode échantillon non
   compressé, G_i = U^{(s)H} Ẑ_i U^{(z)}.
2. Pour LLAE les facteurs sont **complexes** : la reconstruction du code est
   Ẑ_i ≈ U^{(s)} Ĝ_i **U^{(z)H}** (transposée conjuguée), pas U^{(z)⊤} (l. 284 et l. 292) ; la
   formule élément par élément (l. 280) doit porter un conjugué sur U^{(z)} (et la projection
   sur U^{(s)}). Le docstring de `tucker.py` note explicitement que la forme U_s G U_zᵀ de
   l'article n'est exacte que pour le cas réel (SLAE).
3. « orthogonal factor matrices » → pour des matrices complexes, dire « à colonnes orthonormées »
   (semi-unitaires).

### 1.5 Inversion de Tikhonov : le code étend les conjugués avant les équations normales
**STATUT : corrigé** — §2.1 définit maintenant l'ensemble conjugué-symétrique S (eq. `eq:sfull`),
F ∈ C^{K_full×Nt} y est assemblée partout (§2.1, §2.5.2, annexe D), le doublement gratuit des
points est explicité, et la phrase sur Re(·) est reformulée.

**Où :** §2.1 (l. 107–118), Théorème 2.1 et §2.5.2, Annexe D.

**Code :** `laplace_transform/inverse.py` l. 48–56, `learnable.py::_build_inv_matrices`,
`laplace_opti.py::_compute_laplace_matrices` : les fréquences avec Im(s) > 0 sont **explicitement
étendues par leurs conjuguées** (F_full = [F ; conj(F₊)]) avant de former
A = Re(F_fullᴴ F_full) + α_t DᵀD + λI et le second membre.

Conséquences pour l'article :
- La phrase « the conjugate frequencies are implicitly accounted for through the real-part
  operator Re(·) » (l. 118) est inexacte : Re(FᴴF) seul (eq. 5) ne rend PAS compte des
  conjuguées ; le code les ajoute explicitement, ce qui **double le poids des fréquences
  ω_k > 0** par rapport à la fréquence réelle (k = 0) dans le problème (4)–(5) tel qu'écrit.
- Le problème de moindres carrés (4) et les équations normales (5) devraient être écrits avec
  F_full (ou de façon équivalente avec un poids 2 sur les fréquences ω_k > 0), sinon la formule
  ne correspond pas à ce qui est résolu.
- Idem pour toute la section bias–variance (A, facteur d'amplification ‖A⁻¹Fᴴ‖) et sa preuve :
  remplacer F par F_full pour être cohérent avec l'implémentation.

### 1.6 L'optimisation offline du contour (laplace_opti.py) ne minimise pas la loss proxy (14) de l'article
**STATUT : corrigé** — points 1 à 5 traités : termes de régularité (temporel + les DEUX axes
spatiaux) déjà dans eq. `eq:bias_smooth`, résidu SVD réécrit en base unique partagée sur les
fréquences empilées/réalifiées (eq. `eq:svd_residual`, + calcul par matrice de Gram et rang
plafonné à min(d_z, N_opt)), norme de Frobenius assumée dans la borne et l'annexe D,
N_opt = 100 instances à un seul angle, contraintes de boîte et hyperparamètres dans §3.8.
Reste ouvert : le proxy « par fréquence » (celui qui bornerait vraiment la classe de SLAE)
n'est pas celui utilisé — c'est désormais dit explicitement en annexe D plutôt que corrigé.

**Où :** §2.5.2, eq. (13)–(14) (l. 355–396).

**Code :** `scripts/laplace_opti.py`. Divergences :
1. **Terme de biais** : le code minimise
   (‖err‖ + 0.5·‖Δ_t err‖ + 0.5·‖Δ_x err‖ + ‖Δ_y err‖)/(1+0.5+0.5) — il y a des termes de
   régularité **temporelle et spatiale de l'erreur** (λ_diff = λ_x = 0.5) totalement absents de
   l'eq. (14).
2. **Résidu SVD** : l'article écrit Σ_k Σ_{j>d_z} (σ_j^{(k)})² — une SVD tronquée de rang d_z
   **par fréquence**. Le code (`ae_error_lowmem`) somme les matrices de Gram **sur toutes les
   fréquences** (G_global = Σ_k G_k) puis tronque au rang d_z **global** : c'est le résidu d'une
   seule base de rang d_z partagée entre les K fréquences (concaténées), pas la somme des
   résidus par fréquence. Or SLAE encode chaque fréquence séparément en d_z — le proxy « par
   fréquence » de l'article serait le bon ; celui du code est différent. Aligner texte et code
   (ou corriger le code plus tard, mais l'article doit décrire ce qui a été fait).
3. **Norme du facteur d'amplification** : le théorème utilise la norme d'opérateur ; le code
   calcule une norme de **Frobenius** de [A⁻¹Re(F_fullᴴ) ; A⁻¹Im(F_fullᴴ)] (majorant). À préciser.
4. **Données** : E_μ n'est pas prise sur N_train = 6480 mais sur **100 simulations tirées du
   train, à angle de rotation 0 uniquement** (`n_cases=100`, `angle0_idx`). À dire explicitement.
5. **Contraintes/paramètres non mentionnés** : Re(s_k) clampé ≥ −0.05 (des parties réelles
   **négatives** sont autorisées, alors que §2.1 impose γ > 0), Im(s_k) ≥ 0, λ, α_t ∈ [10⁻⁶, 1]
   optimisés directement (init 10⁻³, pas de log-paramétrisation ici), λ_AE = 1.25, AdamW lr 5e-3,
   500 époques. Donner ces valeurs (ou au moins λ_AE et les contraintes).

### 1.7 « Learnable contour » : les résultats LLAE présentés utilisent un contour FIXE
**Où :** contribution 3 (l. 90) — « for LLAE, the Laplace poles are learned end-to-end alongside
the surrogate » ; §2.5.1 (l. 309) — « This is precisely what is done in our LLAE implementation ».

**Code / checkpoints :** `learnable_laplace: false` par défaut (`configs/model/llae.yaml`) ;
les checkpoints des tables de résultats (`llae_ld*_K*_g*` sans suffixe) sont à contour de
Bromwich **fixe**. Les variantes à pôles appris sont des checkpoints séparés suffixés `_ll`
(llae_ld64_K{8,16,32}_g0.0_ll + surrogates hérités), dont les résultats ne figurent pas encore
dans l'article (§3.8 vide). De même les variantes SLAE à contour optimisé sont les `_ol`.

**Correction :** reformuler la contribution et le §2.5.1 pour dire que les pôles *peuvent* être
appris (variante `_ll`, évaluée en §3.8), et que **tous les résultats des §3.3–3.6 utilisent le
contour de Bromwich tronqué fixe**. Sinon le lecteur croit que toutes les LLAE des tables ont des
pôles appris. Préciser aussi (cohérent avec `train_surrogate.py` l. 86–89) que lorsqu'un AE `_ll`
est utilisé, le surrogate hérite des pôles et **continue de les raffiner** avec un lr dédié
(`lr_laplace = 1e-5` — valeur absente de l'annexe).

---

## 2. Formules qui ne correspondent pas à l'implémentation

### 2.1 Toutes les losses sont des MSE (moyennes), pas des normes carrées sommées
**Où :** eq. SLAE (l. 191), eq. LLAE (l. 206–209), loss surrogate (l. 270), loss SVD (l. 149).

**Code :** `F.mse_loss` et `.mean()` partout (`slae.py::loss`, `llae.py::loss`,
`*_surrogate.py::loss`). Les formules de l'article normalisent par 1/N_t ou 1/K mais gardent des
normes ‖·‖² non divisées par N_x ou d_z. Les poids relatifs impliqués diffèrent donc du code par
de gros facteurs (ex. LLAE : le terme de reconstruction porte un facteur N_x ≈ 16 384 par rapport
au terme ridge en d_z si on lit l'article littéralement). Avec les valeurs β = 10⁻², β_latent = 0.1,
α_lat = α_spat = 1 données en annexe, **seule l'écriture en moyennes (MSE) est cohérente**.
Réécrire chaque loss avec des moyennes sur tous les indices (batch, temps/fréquence, espace/latent).

### 2.2 Les champs et latents sont normalisés — jamais mentionné
**Code :** `dataset.fit` : U standardisé pixel par pixel (moyenne/écart-type train) ;
frames de Laplace SLAE standardisées **par fréquence et par pixel** (`lap_mean/lap_std`,
`dataset._compute_laplace`) ; cibles G des surrogates SVD/Tucker standardisées (moyenne/std).
Les losses (SLAE eq. 8, surrogates) opèrent dans ces espaces normalisés. À préciser dans le texte
(au minimum dans l'annexe C) : cela change l'interprétation des erreurs par fréquence (§3.3) et
des poids de loss.

### 2.3 FiLM : le gain passe par un tanh
**Où :** eq. (7) (l. 176).
**Code :** `encoder_decoder.py::_film` et `slae.py` : γ = tanh(W e(τ)), donc modulation
x·(1 + tanh(α)) + δ, bornée dans (0, 2). Ajouter le tanh dans l'eq. (7) (c'est lui qui garantit
l'initialisation proche de l'identité de façon stable).

### 2.4 Forward « FFT »
**Où :** l. 126–130 (« the forward sum reduces to a truncated DFT … making the forward pass
computationally efficient »).
**Code :** `forward.py` et `learnable.py` implémentent la somme par **produit matriciel dense**,
pas par FFT. L'identité mathématique est vraie ; la phrase sur l'efficacité suggère une
implémentation FFT qui n'existe pas. Reformuler (« could be evaluated by FFT ») ou assumer le matmul
(coût O(K·N_t) par pixel, négligeable ici).

### 2.5 Conditionnement du surrogate MLP
**Code :** `surrogate_base.py` : l'encodage sinusoïdal des fréquences du surrogate utilise
**L = 6** niveaux (config `freq_L: 6`) et **pas de MLP** derrière (sin/cos bruts concaténés au
trunk), contrairement à l'encodage L = 8 + MLP des AE (eq. 6). L'annexe C ne donne que « L = 8 »
(section AE) ; ajouter freq_L = 6 du surrogate dans la table C.2 pour éviter la confusion.

---

## 3. Affirmations chiffrées à corriger

1. **« a factor of four below the linear SVD baseline » (§3.3, l. 773)** : 2.92 % vs 7.04 %
   (SVD à k_svd = 128, K = 16) donne un facteur ≈ 2.4, et vs 8.09 % (k_svd = 64) ≈ 2.8.
   « Factor of four » est faux — écrire ≈ 2.4× (ou reformuler en points de pourcentage).
2. **« beyond K = 16, additional modes add noise rather than information » (§3.2, l. 673)** :
   la table 1 montre que K = 32 **améliore** la médiane SVD (7.57 % vs 8.09 % à K = 16 ; seul le
   p90 se dégrade : 15.08 vs 14.67). Nuancer (gain marginal en médiane, queue plus lourde),
   la phrase actuelle contredit la table.
3. **Batch size SLAE = 256 (annexe C, l. 1123 et table C.1)** : la config du dépôt
   (`training/ae.yaml`) est `batch_size: 16` pour les deux familles. Si 256 vient d'un override
   des runs, préciser aussi que pour SLAE l'unité de batch est la **paire (simulation, fréquence)**
   (dataset `_LaplaceFlatDataset`), pas la simulation — sinon les deux « batch sizes » ne sont pas
   comparables. À vérifier sur wandb.
4. **lr surrogate 3e-4 « fixed across all runs » (caption table C.2)** : les variantes SVD et
   Tucker utilisent `lr_surrogate: 5e-4` (configs `surrogate_*_svd.yaml`, `surrogate_*_tucker.yaml`),
   pas 3e-4. Restreindre la caption aux 14 surrogates directs, ou donner les deux valeurs.
5. **Table AE+SVD (l. 935, 950)** : `\multirow{14}` pour 13 lignes LLAE et `\multirow{12}` pour
   13 lignes SLAE (13 + 13 = 26, cohérent avec le texte). Corriger les multirow. Noter aussi
   l'absence de la config LLAE d_z=32/k_svd=32 (le §3.6 sur l'impact de k_svd l'exclut de fait —
   les fourchettes +1.06/+1.94 pp et +0.33/+0.93 pp recalculées depuis la table sont correctes).
6. **« 6480 training simulations » (§3.1 et annexe C)** : en réalité 6480 non-test, divisés
   80/20 en **5184 train / 1296 validation** (`train_val_split: 0.8`) pour tous les modèles
   neuronaux ; seules les bases SVD (learn_svd.py) sont fittées sur les 6480. Préciser, et
   mentionner l'early stopping/sélection de checkpoint sur validation (val/loss pour les AE,
   val/l2rel pour les surrogates).

---

## 4. Précisions à ajouter (reproductibilité / rigueur)

- **Détails d'entraînement absents de l'annexe C** : scheduler ReduceLROnPlateau
  (facteur 0.5, patience 15, min 1e-6), gradient clipping 1.0, précision mixte 16 bits,
  `lr_laplace = 1e-5` pour les runs `_ll`, `lr_V = 5e-4` pour les variantes SVD.
- **γ = 0 dans les résultats vs « γ > 0 » (l. 123–125)** : l'eq. (6) impose γ > 0 mais les tables
  incluent γ = 0 (contour purement imaginaire = DFT tronquée) et l'optimisation offline autorise
  même Re(s) < 0. Assouplir la contrainte dans le texte (γ ≥ 0).
- **Normalisation temporelle du conditionnement** : le code utilise t/(N_t − 1) (ratio dans
  [0, 1]), l'article écrit t/T. Détail, mais autant être exact.
- **SLAE-SVD** : préciser que la base V est **unique et partagée entre fréquences**, fittée sur
  les latents z_k poolés sur (simulations × fréquences) (`slae_svd_surrogate.py` docstring), afin
  que la phrase du §2.4.3 (« collapses d_z independently at each frequency ») ne laisse pas croire
  à une base par fréquence.
- **Solveur** : le texte cite `torch.linalg.solve` (l. 118) ; `LearnableLaplace` (utilisé par LLAE
  et tous les surrogates) résout par **Cholesky** (`cholesky_solve`), `laplace_opti.py` par LU.
  Anecdotique, mais si on cite l'implémentation autant être juste.
- **Décodeur BN vs GN** : le décodeur SLAE (`LaplaceDecoder`) utilise BatchNorm, le décodeur
  LLAE (`ConvDecoder`) GroupNorm par défaut (BN pour les anciens checkpoints — commit cf31a5c).
  Si les checkpoints des tables mélangent BN/GN, le mentionner (ou uniformiser avant publication).

---

## 5. Sections vides / travail restant (pas des erreurs, mais à ne pas oublier)

- §3.7 « Tucker surrogate performance » (l. 1005) : **vide**, alors que le tableau
  récapitulatif (l. 454) annonce « nine pipelines studied » et que les checkpoints Tucker
  existent (r_s ∈ {4, 8, 16} × r_z ∈ {8, 16, 32} sur slae/llae_ld64_K16_g0.01 — nb : la config
  par défaut du dépôt est r_s = 8, r_z = 16).
- §3.8 « Optimal laplace contour choice » (l. 1007) : **vide** — c'est là que doivent aller les
  résultats `_ll` (LLAE pôles appris, K ∈ {8, 16, 32}, g0.0) et `_ol` (SLAE contour optimisé
  offline via laplace_opti_K{8,16,32}.pt, avec α_t/λ rechargés depuis le même fichier —
  cf. `ckpt_utils.laplace_reg`, cohérent avec §1.6.5 ci-dessus).
- §4 entier (Comparison and Discussion) : sous-sections vides.
- Une fois §3.7/3.8 écrits, vérifier que la cohérence `_ll`/`_ol` y est décrite comme dans le
  code : pour `_ll`, pôles + α_t + λ appris (log-paramétrisation) en phase 1 **et** raffinés en
  phase 2 ; pour `_ol`, pôles et régularisation figés issus de `laplace_opti.py`, utilisés à la
  fois pour construire le dataset Laplace (datamodule), pour l'inversion du surrogate et à
  l'inférence.

---

## 6. Note annexe (code, pas article)

`ae_module.py::on_validation_epoch_end` (l. 121) appelle `self.model.laplace.log_dict(...)`
mais `LearnableLaplace` n'expose que `log_scatter`/`log_text` — un run LLAE `_ll` avec logger
wandb planterait ici. Sans impact sur l'article, mais à corriger avant de relancer des runs `_ll`
pour la §3.8.
