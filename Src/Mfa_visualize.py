"""
visualize.py
────────────
Generates all charts and visualizations for the Manufacturing Failure Analysis.

Charts produced:
  1. failure_distribution.png   — Failure vs No Failure count
  2. failure_by_type.png        — Breakdown of failure types
  3. correlation_heatmap.png    — Feature correlation matrix
  4. feature_importance.png     — Top features causing failures
  5. failure_over_time.png      — Failure trend over time

Run directly to generate all charts:
    python src/visualize.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (
    PROCESSED_PATH, CHARTS_DIR, MODEL_PATH,
    TARGET_COL, FAILURE_TYPE, TIME_COL, FEATURE_COLS,
    PRIMARY_COLOR, SECONDARY_COLOR, DANGER_COLOR,
    CHART_DPI, CHART_STYLE
)


def setup():
    """Set chart style and create output directory."""
    plt.style.use(CHART_STYLE)
    os.makedirs(CHARTS_DIR, exist_ok=True)


def save_chart(filename: str):
    """Save the current figure to the charts directory."""
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✅ Saved → {path}")


def plot_failure_distribution(df: pd.DataFrame):
    """
    Bar chart showing count of failures vs normal operations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')

    counts = df[TARGET_COL].value_counts()
    labels = ['No Failure', 'Failure']
    colors = [PRIMARY_COLOR, DANGER_COLOR]

    # Bar chart
    axes[0].set_facecolor('#161b22')
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor='#30363d')
    axes[0].set_title('Failure Distribution', color='white', fontsize=14, pad=15)
    axes[0].set_ylabel('Count', color='#8b949e')
    axes[0].tick_params(colors='#8b949e')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     str(val), ha='center', color='white', fontweight='bold')

    # Pie chart
    axes[1].set_facecolor('#161b22')
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
    )
    for text in texts + autotexts:
        text.set_color('white')
    axes[1].set_title('Failure Rate (%)', color='white', fontsize=14, pad=15)

    plt.suptitle('Manufacturing Failure Distribution', color='white', fontsize=16, y=1.02)
    save_chart('failure_distribution.png')


def plot_failure_by_type(df: pd.DataFrame):
    """
    Horizontal bar chart showing breakdown of failure types.
    """
    if FAILURE_TYPE not in df.columns:
        print(f"  ⚠️  Column '{FAILURE_TYPE}' not found, skipping failure type chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # Only show rows where failure occurred
    failures = df[df[TARGET_COL] == 1][FAILURE_TYPE].value_counts()

    colors = [PRIMARY_COLOR, SECONDARY_COLOR, DANGER_COLOR, '#f78166', '#6e40c9']
    bars = ax.barh(failures.index, failures.values,
                   color=colors[:len(failures)], edgecolor='#30363d')

    ax.set_title('Failures by Type', color='white', fontsize=14, pad=15)
    ax.set_xlabel('Number of Failures', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, failures.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='white', fontweight='bold')

    save_chart('failure_by_type.png')


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Heatmap showing correlations between all numerical features.
    """
    existing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    corr = df[existing_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='RdYlGn',
        center=0, ax=ax, linewidths=0.5, linecolor='#0d1117',
        annot_kws={'size': 10, 'color': 'white'}
    )
    ax.set_title('Feature Correlation Heatmap', color='white', fontsize=14, pad=15)
    ax.tick_params(colors='#8b949e', labelsize=9)

    save_chart('correlation_heatmap.png')


def plot_feature_importance(df: pd.DataFrame):
    """
    Horizontal bar chart of feature importances from the trained model.
    """
    if not os.path.exists(MODEL_PATH):
        print("  ⚠️  Model not found. Run train.py first for feature importance chart.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    existing_cols = [c for c in FEATURE_COLS if c in df.columns]

    if not hasattr(model, 'feature_importances_'):
        print("  ⚠️  Model does not support feature importances.")
        return

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'feature': existing_cols,
        'importance': importances[:len(existing_cols)]
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    colors = [DANGER_COLOR if i == feat_df['importance'].idxmax()
              else PRIMARY_COLOR for i in feat_df.index]

    bars = ax.barh(feat_df['feature'], feat_df['importance'],
                   color=colors, edgecolor='#30363d')

    ax.set_title('Feature Importance — What Causes Failures?',
                 color='white', fontsize=14, pad=15)
    ax.set_xlabel('Importance Score', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = [
        mpatches.Patch(color=DANGER_COLOR, label='Most Important'),
        mpatches.Patch(color=PRIMARY_COLOR, label='Other Features')
    ]
    ax.legend(handles=legend, facecolor='#161b22',
              labelcolor='white', edgecolor='#30363d')

    save_chart('feature_importance.png')


def plot_failure_over_time(df: pd.DataFrame):
    """
    Line chart showing number of failures over time.
    """
    if TIME_COL not in df.columns:
        print(f"  ⚠️  Column '{TIME_COL}' not found, skipping time chart")
        return

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    monthly = df.groupby(df[TIME_COL].dt.to_period('M'))[TARGET_COL].sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ax.plot(monthly.index.astype(str), monthly.values,
            color=DANGER_COLOR, linewidth=2.5, marker='o',
            markersize=6, markerfacecolor='white')
    ax.fill_between(range(len(monthly)), monthly.values,
                    alpha=0.15, color=DANGER_COLOR)

    ax.set_title('Failure Frequency Over Time', color='white', fontsize=14, pad=15)
    ax.set_xlabel('Month', color='#8b949e')
    ax.set_ylabel('Number of Failures', color='#8b949e')
    ax.tick_params(colors='#8b949e', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_chart('failure_over_time.png')


def generate_all_charts():
    """Load data and generate all visualizations."""
    print("\n" + "═" * 50)
    print("   GENERATING VISUALIZATIONS")
    print("═" * 50)

    setup()

    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed data not found: {PROCESSED_PATH}\n"
            "Please run preprocess.py first."
        )

    df = pd.read_csv(PROCESSED_PATH)
    print(f"\n  Loaded {len(df)} rows for visualization\n")

    print("[1/5] Failure distribution...")
    plot_failure_distribution(df)

    print("[2/5] Failure by type...")
    plot_failure_by_type(df)

    print("[3/5] Correlation heatmap...")
    plot_correlation_heatmap(df)

    print("[4/5] Feature importance...")
    plot_feature_importance(df)

    print("[5/5] Failure over time...")
    plot_failure_over_time(df)

    print(f"\n✅ All charts saved to {CHARTS_DIR}\n")


if __name__ == '__main__':
    generate_all_charts()
