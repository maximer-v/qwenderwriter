"""Model-to-tensor compiler: manifest schema, normalized tree IR, exact tensor modules, receipts and parity."""
from .compiler import compile_manifest, load_compiled, save_compiled
from .ir import TreeEnsembleIR, XGBoostJSONAdapter
from .schema import CompiledManifest, ManifestError

__all__ = ["CompiledManifest", "ManifestError", "TreeEnsembleIR", "XGBoostJSONAdapter", "compile_manifest", "save_compiled", "load_compiled"]
