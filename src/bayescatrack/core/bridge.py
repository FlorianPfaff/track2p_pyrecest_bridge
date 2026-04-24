"""Compatibility bridge for the BayesCaTrack package layout."""

import track2p_pyrecest_bridge as _bridge

from .._exports import BRIDGE_PUBLIC_NAMES, reexport

__all__ = reexport(_bridge, globals(), BRIDGE_PUBLIC_NAMES)
