"""Compatibility shim for the unified training API and frontend overlays."""

from mikazuki.app.training_api import router
from mikazuki.frontend_branding import install_frontend_branding_patch

# training_api installs the effective-config frontend patch during import. Apply
# project branding afterwards so layout/app patches compose in a deterministic
# order without modifying the pinned frontend submodule.
install_frontend_branding_patch()
