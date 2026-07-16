"""
dl_rom — Baseline DL-ROM (Fresca, Dedè, Manzoni 2021), sans transformée de Laplace.

Mêmes briques que le pipeline LLAE (ConvEncoder/ConvDecoder + FreqSurrogate,
même latent_dim, même protocole en deux phases), mais le surrogate prédit
directement z(t) pour les Nt pas de temps : la seule différence avec le LLAE
est l'absence du bottleneck de Laplace.
"""
from dl_rom.dlrom_ae import DLROMAE
from dl_rom.dlrom_surrogate import DLROMModel

__all__ = ['DLROMAE', 'DLROMModel']
