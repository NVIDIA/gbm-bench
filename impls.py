class ImplementationPlaceHolder:
    __version__ = None


try:
    import xgboost
except ImportError:
    xgboost = ImplementationPlaceHolder()
xgb = xgboost

try:
    import lightgbm
except ImportError:
    lightgbm = ImplementationPlaceHolder()
lgb = lightgbm

try:
    import catboost
except ImportError:
    catboost = ImplementationPlaceHolder()
cat = catboost

try:
    import dask_xgboost
except ImportError:
    dask_xgboost = ImplementationPlaceHolder()
dxgb = dask_xgboost
