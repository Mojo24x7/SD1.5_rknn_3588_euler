"""
Auto-loaded by Python at startup (if on sys.path).

Goal:
- diffusers newer versions may reference torch.xpu.* and torch.distributed.device_mesh
  even on systems that don't have XPU or newer distributed features.
- torch 2.2.0 on aarch64 may not expose these attributes.
We patch torch *after it is imported* (via import hook), so it works reliably.
"""

import builtins
import types

def _patch_torch(torch):
    # ---- torch.xpu shim ----
    if not hasattr(torch, "xpu"):
        class _DummyXPU:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def device_count(): return 0
            @staticmethod
            def current_device(): return 0
            @staticmethod
            def synchronize(device=None): return None
            @staticmethod
            def empty_cache(): return None
            @staticmethod
            def manual_seed(seed): return None
            @staticmethod
            def manual_seed_all(seed): return None

            # catch anything else diffusers might touch
            def __getattr__(self, name):
                def _noop(*args, **kwargs):
                    return None
                return _noop

        # IMPORTANT: set attribute directly on module so hasattr() becomes True
        torch.__dict__["xpu"] = _DummyXPU()

    # ---- torch.distributed.device_mesh shim ----
    try:
        dist = getattr(torch, "distributed", None)
        if dist is not None and not hasattr(dist, "device_mesh"):
            # minimal placeholder for type annotations / attribute access
            dist.device_mesh = types.SimpleNamespace(DeviceMesh=object)
    except Exception:
        pass


# Import hook: patch torch right after it's imported anywhere
_orig_import = builtins.__import__

def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    if name == "torch" or name.startswith("torch"):
        try:
            import torch as _t
            _patch_torch(_t)
        except Exception:
            pass
    return mod

builtins.__import__ = _hooked_import
