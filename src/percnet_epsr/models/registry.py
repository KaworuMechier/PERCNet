"""Model construction and the frozen per-model training recipes."""

from __future__ import annotations

from types import SimpleNamespace

import torch


# Unified training recipe, taken from the reproduced FESTRNN protocol:
#   Adam(lr=8e-4, weight_decay=1e-5), grad clip 5, 50 epochs,
#   ReduceLROnPlateau(factor=0.5, patience=2) on validation MAPE.
# Model-specific losses stay model-specific; only the optimiser/schedule/
# budget are shared.  Selection remains protocol-dependent:
#   official = best test epoch (reproduction anchor only)
#   honest   = best validation epoch (main tables)
UNIFIED_RECIPE = dict(lr=8e-4, weightdecay=1e-5, clip=5.0, epochs=50,
                      scheduler="plateau", plateau_factor=0.5,
                      plateau_patience=2, early_stop=False)

# Explicit v3 variant table: each entry switches one component off (or on),
# so the ablation table never depends on substring parsing of the name.
V3_VARIANTS = {
    "percnet_v3": {},
    "percnet_v3_noexo": {"use_exo_channels": False},
    "percnet_v3_nofusion": {"fusion_head": False},
    "percnet_v3_nodeg": {"degree_response": False},
    "percnet_v3_noeesm": {"use_eesm": False},
    # 3 load channels and no fusion: the v2 structure with v3's dropout
    "percnet_v3_plain": {"use_exo_channels": False, "fusion_head": False},
    # pure point objective: no quantile/scale head, MAPE on mu only
    "percnet_v3_point": {"use_exo_channels": False, "prob_head": False},
    # recent observed weather as fusion context instead of extra streams
    "percnet_v3_wx": {"use_exo_channels": False, "weather_context": True},
    # same, plus a trailing 72 h window for the multi-day thermal history
    "percnet_v3_wx72": {"use_exo_channels": False, "weather_context": True,
                        "wx_windows": (24, 72)},
    # weather context with the pure point objective
    "percnet_v3_wx_point": {"use_exo_channels": False, "weather_context": True,
                            "prob_head": False},
    # weather context also conditioning the quantile fan / scale head
    "percnet_v3_wxq": {"use_exo_channels": False, "weather_context": True,
                       "wx_quantile": True},
    # context restricted to the six true weather channels (cols 3-8).  The
    # default n_exo=9 also swallowed CMBMBBTU and GHG, which are derived from
    # the targets, so this cell separates weather information from a
    # load-derived shortcut.
    "percnet_v3_wxw": {"use_exo_channels": False, "weather_context": True,
                       "n_exo": 6},
    # decomposition-loss weight removed (was 10.0)
    "percnet_v3_wx_ndec": {"use_exo_channels": False, "weather_context": True,
                           "lam_dec": 0.0},
    # carried-over priors from the previous version, ablated one at a time
    "percnet_v3_wx_nocyc": {"use_exo_channels": False, "weather_context": True,
                            "use_cycle_queue": False},
    "percnet_v3_wx_noanc": {"use_exo_channels": False, "weather_context": True,
                            "use_last_value_anchor": False},
    # weather context at the reference architecture's hidden width
    "percnet_v3_wxh64": {"use_exo_channels": False, "weather_context": True,
                         "linear_hidden": 64},
}


MODEL_RECIPES = {
    "percnet_v2": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v2_eesm": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v2_ema": dict(UNIFIED_RECIPE, loss_type="model"),
    # v3 family: one full model plus the ablation switches (see percnet_v3.py)
    "percnet_v3": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_noexo": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_nofusion": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_nodeg": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_noeesm": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_plain": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_point": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wx": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wx72": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wx_point": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wxq": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wxw": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wx_ndec": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wx_nocyc": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wx_noanc": dict(UNIFIED_RECIPE, loss_type="model"),
    "percnet_v3_wxh64": dict(UNIFIED_RECIPE, loss_type="model"),
    "dlinear": dict(UNIFIED_RECIPE, loss_type="swl1"),
    "itransformer": dict(UNIFIED_RECIPE, loss_type="swl1"),
    "segrnn": dict(UNIFIED_RECIPE, loss_type="swl1"),
    "crossformer": dict(UNIFIED_RECIPE, loss_type="swl1"),
    "film": dict(UNIFIED_RECIPE, loss_type="swl1"),
    "mamba": dict(UNIFIED_RECIPE, loss_type="swl1"),
    "tft": dict(UNIFIED_RECIPE, loss_type="pinball"),
    "deepar": dict(UNIFIED_RECIPE, loss_type="nll"),
    "deeparq": dict(UNIFIED_RECIPE, loss_type="pinball"),
}


def make_config(model_name: str, horizon: int, spec,
                overrides: dict | None = None) -> SimpleNamespace:
    cfg = SimpleNamespace(
        model=model_name,
        seq_len=int(spec.seq_len),
        pred_len=int(horizon),
        enc_in=int(spec.enc_in),
        n_free=int(spec.n_free),
        n_tgt=int(spec.n_tgt),
        use_calendar=bool(spec.use_calendar),
        physics_identity=bool(spec.physics_identity),
        cel_coef=tuple(spec.cel_coef),
        loss_scale_weighted=True,
        batchsize=64,
    )

    if model_name in ("percnet_v2", "percnet_v2_eesm", "percnet_v2_ema",
                      "percnet_v2_xcpd"):
        cfg.n_quantiles = 9
        cfg.cvar_beta = 0.9
        cfg.lam_pin = 1.0
        cfg.lam_nll = 1.0
        cfg.lam_tail = 0.5
        cfg.lam_mono = 0.1
        cfg.lam_dec = 10.0
        cfg.linear_hidden = 36
        cfg.decomp_kernel = 13
        cfg.prnn_seg_len = 24
        cfg.prnn_dim = 12
        cfg.prnn_dropout = 0.1
        cfg.dropout = 0.1
        cfg.use_last_value_anchor = True
        cfg.use_cycle_queue = True
        cfg.cycle_gate = False
        cfg.prob_head = True
        cfg.prob_detach = False
        cfg.horizon_cond_head = True
        cfg.point_loss = "mape"
        cfg.use_eesm = model_name in ("percnet_v2_eesm",
                                      "percnet_v2_ema",
                                      "percnet_v2_xcpd")
    elif model_name.startswith("percnet_v3"):
        # Same core recipe as v2 plus the full-channel switches; the variant
        # suffix turns one component off for the ablation table.
        cfg.n_quantiles = 9
        cfg.cvar_beta = 0.9
        cfg.lam_pin = 1.0
        cfg.lam_nll = 1.0
        cfg.lam_tail = 0.5
        cfg.lam_mono = 0.1
        cfg.lam_dec = 10.0
        cfg.linear_hidden = 36
        cfg.decomp_kernel = 13
        cfg.prnn_seg_len = 24
        cfg.prnn_dim = 12
        # v3 carries more effective capacity than v2 (12 channels, fusion,
        # degree), so it needs the regularisation the 12-channel reference
        # architecture uses rather than v2's 0.1.
        cfg.prnn_dropout = 0.3
        cfg.dropout = 0.1
        cfg.use_last_value_anchor = True
        cfg.use_cycle_queue = True
        cfg.cycle_gate = False
        cfg.prob_head = True
        cfg.prob_detach = False
        cfg.horizon_cond_head = True
        cfg.point_loss = "mape"
        cfg.n_chan = 12
        cfg.temp_idx = 3
        cfg.degree_breakpoints = 3
        cfg.n_exo = 9
        cfg.wx_windows = (24,)
        cfg.use_exo_channels = True
        cfg.fusion_head = True
        cfg.degree_response = True
        cfg.weather_context = False
        cfg.wx_quantile = False
        cfg.use_eesm = True
        for k, v in V3_VARIANTS.get(model_name, {}).items():
            setattr(cfg, k, v)
    elif model_name == "dlinear":
        cfg.individual = True
        cfg.moving_avg = 25
    elif model_name == "itransformer":
        cfg.d_model, cfg.n_heads = 128, 8
        cfg.n_layers, cfg.d_ff = 2, 256
    elif model_name == "segrnn":
        cfg.seg_len, cfg.d_model, cfg.dropout = 24, 64, 0.1
    elif model_name == "crossformer":
        # reference configuration (FreqST_Lite/models_baselines)
        cfg.seg_count, cfg.d_model, cfg.n_heads, cfg.n_layers = 6, 64, 4, 2
        cfg.dropout = 0.1
    elif model_name == "film":
        cfg.degree, cfg.period, cfg.dropout = 24, 24, 0.1
    elif model_name == "mamba":
        # S-Mamba: variate tokens (length = enc_in), bidirectional scan
        cfg.d_model, cfg.n_layers, cfg.d_state = 64, 2, 16
        cfg.dropout = 0.1
    elif model_name == "tft":
        cfg.hidden, cfg.n_heads = 24, 2
        cfg.dropout = 0.1
        cfg.n_quantiles = 9
        cfg.prob_head = True
    elif model_name in ("deepar", "deeparq"):
        cfg.hidden, cfg.n_layers, cfg.emb_dim = 48, 2, 8
        cfg.dropout = 0.1
        cfg.use_window_scale = True
        cfg.likelihood = "gaussian"
        cfg.sigma_floor = None
        cfg.n_quantiles = 9
        cfg.teacher_forcing = model_name == "deeparq"

    cfg.lr = MODEL_RECIPES.get(model_name, {}).get("lr", 3e-4)
    cfg.weightdecay = MODEL_RECIPES.get(model_name, {}).get(
        "weightdecay", 0.0)
    cfg.clip = MODEL_RECIPES.get(model_name, {}).get("clip", None)
    cfg.epochs = MODEL_RECIPES.get(model_name, {}).get("epochs", 100)
    cfg.scheduler = MODEL_RECIPES.get(model_name, {}).get("scheduler", "cosine")
    cfg.plateau_factor = MODEL_RECIPES.get(model_name, {}).get(
        "plateau_factor", 0.5)
    cfg.plateau_patience = MODEL_RECIPES.get(model_name, {}).get(
        "plateau_patience", 6)
    cfg.early_stop = MODEL_RECIPES.get(model_name, {}).get(
        "early_stop", True)
    cfg.loss_type = MODEL_RECIPES.get(model_name, {}).get(
        "loss_type", "swl1")
    cfg.use_ema = model_name.endswith("_ema")

    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    return cfg


def build_model(model_name: str, cfg):
    if model_name in ("percnet_v2", "percnet_v2_eesm", "percnet_v2_ema"):
        from percnet_epsr.models.percnet_v2 import Model
        return Model(cfg)
    if model_name.startswith("percnet_v3"):
        from percnet_epsr.models.percnet_v3 import Model
        return Model(cfg)
    if model_name == "dlinear":
        from percnet_epsr.models.baselines.dlinear import Model
        return Model(cfg)
    if model_name == "itransformer":
        from percnet_epsr.models.baselines.itransformer import Model
        return Model(cfg)
    if model_name == "segrnn":
        from percnet_epsr.models.baselines.segrnn import Model
        return Model(cfg)
    if model_name == "crossformer":
        from percnet_epsr.models.baselines.crossformer import Model
        return Model(cfg)
    if model_name == "film":
        from percnet_epsr.models.baselines.film import Model
        return Model(cfg)
    if model_name == "mamba":
        from percnet_epsr.models.baselines.mamba import Model
        return Model(cfg)
    if model_name == "tft":
        from percnet_epsr.models.baselines.tft import Model
        return Model(cfg)
    if model_name == "deepar":
        from percnet_epsr.models.baselines.deepar import Model
        return Model(cfg)
    if model_name == "deeparq":
        from percnet_epsr.models.baselines.deeparq import Model
        return Model(cfg)
    raise ValueError(f"unknown model {model_name!r}")


def build_xcpd(cfg, backbone):
    from percnet_epsr.models.xcpd import Model
    return Model(cfg, backbone)
