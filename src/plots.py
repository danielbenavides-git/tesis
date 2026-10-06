import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loss_curves(history, components=("total", "recon", "kl")):
    df = pd.DataFrame(history)
    rows = []
    for comp in components:
        for split in ("train", "val"):
            col = f"{split}_{comp}"
            if col in df:
                rows.append(pd.DataFrame({"epoch": df["epoch"], "value": df[col], "split": split, "component": comp}))
    long = pd.concat(rows)
    charts = [
        alt.Chart(long[long["component"] == comp]).mark_line(point=True).encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color("split:N", title="Split",
                scale=alt.Scale(domain=["train", "val"], range=["#4c78a8", "#f58518"]),
                legend=alt.Legend(orient="top")),
            strokeDash=alt.StrokeDash("split:N", legend=None),
        ).properties(width=250, height=200, title=comp)
        for comp in components
    ]
    return alt.hconcat(*charts)


def selection_chart(table, k_best):
    """BIC (GMM) or inertia (k-means) next to the separation metrics, all on training days. Dashed line marks k_best."""
    base = alt.Chart(table).encode(x=alt.X("k:O", title="Number of regimes (k)"))
    rule = alt.Chart(pd.DataFrame({"k": [k_best]})).mark_rule(strokeDash=[4, 4], color="#e45756").encode(x="k:O")
    criterion = "bic" if "bic" in table else "inertia"
    panels = [
        (criterion, "BIC (lower is better)" if criterion == "bic" else "Inertia (look for the elbow)",
         "GMM BIC" if criterion == "bic" else "K-means inertia", "#4c78a8"),
        ("silhouette", "Silhouette (higher is better)", "Silhouette", "#54a24b"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)", "Davies-Bouldin", "#f58518"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)", "Calinski-Harabasz", "#b279a2"),
    ]
    charts = [
        (base.mark_line(point=alt.OverlayMarkDef(color=color), color=color).encode(y=alt.Y(f"{col}:Q", title=y_title, scale=alt.Scale(zero=False)))
         + rule).properties(width=220, height=200, title=title)
        for col, y_title, title, color in panels if col in table
    ]
    return alt.hconcat(*charts)


def tsne_chart(coords, labels, title):
    df = pd.DataFrame({"tsne_1": coords[:, 0], "tsne_2": coords[:, 1], "regime": labels.astype(str)})
    return alt.Chart(df).mark_circle(size=8, opacity=0.6).encode(
        x=alt.X("tsne_1:Q", title="t-SNE 1"),
        y=alt.Y("tsne_2:Q", title="t-SNE 2"),
        color=alt.Color("regime:N", title="Regime"),
    ).properties(width=450, height=400, title=title)


def _el_nino_periods(context):
    is_nino = (context["enso_phase"] == "El Nino").to_numpy()
    change = np.flatnonzero(np.diff(is_nino.astype(int))) + 1
    bounds = np.concatenate([[0], change, [len(is_nino)]])
    periods = [(context["date"].iloc[a], context["date"].iloc[b - 1]) for a, b in zip(bounds[:-1], bounds[1:]) if is_nino[a]]
    return pd.DataFrame(periods, columns=["start", "end"])


def regime_timeline(context, split_dates=None, title="Detected regimes over time"):
    """Real price colored by regime, regime strip and reservoir level. El Nino periods shaded, split boundaries dashed."""
    df = context.assign(regime=context["regime"].astype(str))
    x = alt.X("date:T", title="")
    nino = alt.Chart(_el_nino_periods(context)).mark_rect(color="#e45756", opacity=0.12).encode(x="start:T", x2="end:T")
    layers_extra = [nino]
    if split_dates:
        rules = pd.DataFrame({"date": pd.to_datetime(split_dates)})
        layers_extra.append(alt.Chart(rules).mark_rule(strokeDash=[4, 4], color="#555555").encode(x="date:T"))

    price = alt.Chart(df).mark_circle(size=6, opacity=0.7).encode(
        x=x,
        y=alt.Y("precio_real_cop_kwh:Q", title="Real price (COP/kWh, log)", scale=alt.Scale(type="log")),
        color=alt.Color("regime:N", title="Regime"),
    )
    strip = alt.Chart(df).mark_tick(thickness=1, size=12).encode(
        x=x, y=alt.Y("regime:O", title="Regime"), color=alt.Color("regime:N", legend=None),
    )
    reservoir = alt.Chart(df).mark_line(strokeWidth=0.6, color="#4c78a8").encode(
        x=x, y=alt.Y("volumen_util_pct:Q", title="Reservoir level"),
    )
    return alt.vconcat(
        alt.layer(*layers_extra, price).properties(width=800, height=260, title=title),
        alt.layer(*layers_extra, strip).properties(width=800, height=100),
        alt.layer(*layers_extra, reservoir).properties(width=800, height=120),
    ).resolve_scale(color="shared")


def plot_reconstructions(originals, reconstructions, labels, n_show=8):
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 5))
    for i in range(n_show):
        for row, img, name in ((0, originals[i], "Original"), (1, reconstructions[i], "Reconstructed")):
            axes[row, i].imshow(img[0], cmap="viridis", aspect="auto", origin="upper", vmin=0, vmax=1)
            axes[row, i].set_title(f"{name}\n{labels[i]}" if row == 0 else name, fontsize=8)
            axes[row, i].axis("off")
    fig.tight_layout()
    return fig
