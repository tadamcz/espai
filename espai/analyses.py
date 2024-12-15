"""
These are pretty ad-hoc and may not be used for long.
Correspondingly, don't expect the code quality to be high.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from espai import Framing, FIELDS, Question
from espai.config import Config
from espai import PROJECT_ROOT
from espai.aggregate import summarize_dists, AggMethod
from espai.fit import df_fit, df_calc_err, df_fitted_values, DistFamily, LossFunction, get_x_p
from espai.preprocess import merged, filter_rename
from typing import List, Optional, Tuple


# Needed to show lines at 0 or 1
CDF_YLIM_BUFFER = 0.025
CDF_YLIM = (-CDF_YLIM_BUFFER, 1 + CDF_YLIM_BUFFER)


def individual_replication():
    for question in [Question.HLMI, Question.FAOL]:
        for framing in [Framing.FY, Framing.FP]:
            print(f"Question: {question.name}, Framing: {framing}")
            data = merged(question)
            data = filter_rename(data, question, framing)
            data = df_fit(data, question, framing, DistFamily.GAMMA, LossFunction.CDF_MSE_PROBS)

            data.loc[:, "mean_sq_err"] = df_calc_err(
                data,
                "fitted_dist",
                question,
                framing,
            )

            def attach_prev_fitted_dist(row):
                return DistFamily.GAMMA.freeze(shape=row["prev_shape"], scale=row["prev_scale"])

            data.loc[:, "prev_fitted_dist"] = data.apply(attach_prev_fitted_dist, axis=1)

            data.loc[:, "prev_mean_sq_err"] = df_calc_err(
                data,
                "prev_fitted_dist",
                question,
                framing,
            )

            err_diff = data["mean_sq_err"] - data["prev_mean_sq_err"]
            print(f"NaNs: {err_diff.isna().sum()} / {len(err_diff)}")
            print(f"Improvements: {sum(err_diff < -1e-6)} / {len(err_diff)}")
            print(f"Degradations: {sum(err_diff > 1e-6)} / {len(err_diff)}")

            data.loc[:, "shape_diff"] = data["shape"] - data["prev_shape"]
            data.loc[:, "scale_diff"] = data["scale"] - data["prev_scale"]

            # Print shape, scale for 10 largest shape_diff and scale_diff
            top_shape_diff = data.nlargest(10, "shape_diff")
            top_scale_diff = data.nlargest(10, "scale_diff")

            cols = [
                "shape",
                "prev_shape",
                "scale",
                "prev_scale",
                *framing.human_columns,
            ]

            with pd.option_context("display.max_columns", None, "display.width", None):
                print(f"Top shape_diff:\n{top_shape_diff[cols]}")
            print(f"Top scale_diff:\n{top_scale_diff[cols]}")


def agg_replication_plots():
    for question in [Question.HLMI, Question.FAOL]:
        print(f"Question: {question.name}")
        data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")
        data = filter_rename(data, question, None)
        data = df_fit(
            data,
            question,
            None,
            DistFamily.GAMMA,
            LossFunction.CDF_MSE_PROBS
            if DistFamily.GAMMA not in [DistFamily.cinterp3, DistFamily.cinterp5]
            else None,
        )
        data = data.dropna(subset=["shape", "scale"])

        previous_aggregated = pd.read_csv(
            PROJECT_ROOT / "data" / "aggregated" / f"{question.name}_2023.csv"
        )

        # Generate years array
        survey_year = 2023

        plot_to_year = 2125

        plot_to_year_rel = plot_to_year - survey_year
        rel_years = np.linspace(0, plot_to_year_rel, 1000)
        # Calculate our aggregated CDF
        our_cdf = summarize_dists(
            data, rel_years, dist_col="fitted_dist", method=AggMethod.ARITH_MEAN_PROBS
        )["aggregated"]

        # Create plot
        plt.figure(figsize=(10, 6))

        previous_aggregated = previous_aggregated[
            previous_aggregated["x"].between(0, plot_to_year_rel)
        ]

        plt.plot(survey_year + previous_aggregated["x"], previous_aggregated["y"], label="Previous")
        plt.plot(survey_year + rel_years, our_cdf, label="Ours")

        # Add labels and title
        plt.xlabel("Years")
        plt.ylabel("Probability")
        plt.title(f"Replication: {question}, arithmetic mean of probs")
        plt.legend()
        plt.xlim(survey_year, plot_to_year)
        plt.show()


def vs_previous(families: List[DistFamily]):
    """
    TODO: refactor repetition with agg_replication_plots
    """
    for question in [Question.HLMI, Question.FAOL]:
        for family in families:
            print(f"Question: {question.name}, Family: {family}")
            data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")
            data = filter_rename(data, question, None)

            loss_function = (
                None
                if family in [DistFamily.cinterp3, DistFamily.cinterp5]
                else LossFunction.CDF_MSE_PROBS
            )
            data = df_fit(data, question, None, family, loss_function)
            data = data.dropna(subset=["fitted_dist"])

            previous_aggregated = pd.read_csv(
                PROJECT_ROOT / "data" / "aggregated" / f"{question.name}_2023.csv"
            )

            # Generate years array
            survey_year = 2023

            plot_to_year = 2100

            plot_to_year_rel = plot_to_year - survey_year
            rel_years = np.linspace(0, plot_to_year_rel, 1000)
            # Calculate our aggregated CDF
            our_cdf = summarize_dists(
                data, rel_years, dist_col="fitted_dist", method=AggMethod.ARITH_MEAN_PROBS
            )["aggregated"]

            # Create plot
            plt.figure(figsize=(10, 6))

            previous_aggregated = previous_aggregated[
                previous_aggregated["x"].between(0, plot_to_year_rel)
            ]

            plt.plot(
                survey_year + previous_aggregated["x"],
                previous_aggregated["y"],
                label="Previous (Gamma)",
            )
            plt.plot(survey_year + rel_years, our_cdf, label=f"Flexible ({family})")

            # Add labels and title
            plt.xlabel("Years")
            plt.ylabel("Probability")
            plt.title(f"{question}, arithmetic mean of probs")
            plt.legend()
            plt.xlim(survey_year, plot_to_year)
            plt.show()


def prev_fits_bias_scatter():
    for question in [Question.HLMI, Question.FAOL]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{question.name}")

        for row, framing in enumerate([Framing.FY, Framing.FP]):
            data = merged(question)
            data = filter_rename(data, question, framing)

            cdf_data = data[framing.human_columns]

            if framing == Framing.FY:
                # Source data expresses probabilities as values between 0 and 100
                cdf_data = cdf_data / 100

            fitted_data = df_fitted_values(
                data,
                {"shape": "prev_shape", "scale": "prev_scale"},
                question,
                framing,
                DistFamily.GAMMA,
            )

            if framing == Framing.FY:
                col_unit = "Probability"
            elif framing == Framing.FP:
                col_unit = "Years"

            for i, col in enumerate(framing.human_columns):
                ax = axs[row, i]
                ax.scatter(cdf_data[col], fitted_data[f"{col}_fitted"], alpha=0.5)
                ax.set_xlabel(f"Actual {col} ({col_unit})")
                ax.set_ylabel(f"Fitted {col} ({col_unit})")
                ax.set_title(f"{framing}, {col}")

                ax.plot(
                    [min(cdf_data[col]), max(cdf_data[col])],
                    [min(cdf_data[col]), max(cdf_data[col])],
                    "r",
                )

                if framing == Framing.FP:
                    ax.set_xlim(0, 1000)
                    ax.set_ylim(0, 1000)

        plt.tight_layout()
        plt.show()


def plot_individual_cdf(ax, row, question, framing, dist_families, colors):
    """
    Plot a single CDF on the given axis.

    Args:
    - ax: The matplotlib axis to plot on
    - row: The data row containing the fitted distributions
    - question: The question being analyzed
    - framing: The framing being used (FY or FP)
    - dist_families: List of DistFamily types to plot
    - colors: List of colors for each distribution family
    """
    # Plot source data as scatter
    x_values, y_values = get_x_p(row, question, framing)
    ax.scatter(x_values, y_values, color="blue", alpha=0.6, label="Source Data")

    # Plot fitted distributions as curves
    if framing == Framing.FY:
        x_range = np.linspace(0, 100, 1000)
    elif framing == Framing.FP:
        x_range = np.linspace(-0.1, max(x_values) * 1.2, 1000)

    for dist_family, color in zip(dist_families, colors):
        fitted_dist = row[f"fitted_dist_{dist_family.name}"]
        y_fitted = fitted_dist.cdf(x_range)
        ax.plot(x_range, y_fitted, color=color, label=f"{dist_family.name} CDF")

    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Probability")
    ax.legend(fontsize="x-small")

    if framing == Framing.FY:
        ax.set_xlim(-0.1, 100)
    elif framing == Framing.FP:
        ax.set_xlim(-0.1, max(x_values) * 1.2)
    ax.set_ylim(CDF_YLIM)


def plot_top_cdfs(
    question: Question,
    framing: Framing,
    dist_families: List[DistFamily],
    n: int = 60,
    sort_by: Optional[Tuple[DistFamily, ...]] = None,
):
    """
    Plot n CDFs with the highest mean square error or difference in MSE, showing source data as scatter
    and fitted distributions as curves for multiple DistFamily types.

    Args:
    - question: The question to analyze
    - framing: The framing to use (FY or FP)
    - dist_families: List of DistFamily types to fit and plot
    - n: Number of CDFs to plot
    - sort_by: Optional sorting criterion. Can be:
               - None: No sorting
               - A 1-tuple with a single DistFamily: Sort by highest MSE for that family
               - A 2-tuple with two DistFamily objects: Sort by highest difference in MSE between the two families
    """
    data = merged(question)
    data = filter_rename(data, question, framing)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(dist_families)))

    # Fit data for each distribution family
    for dist_family in dist_families:
        loss_function = (
            None
            if dist_family in [DistFamily.cinterp3, DistFamily.cinterp5]
            else LossFunction.CDF_MSE_PROBS
        )
        fitted_data = df_fit(data.copy(), question, framing, dist_family, loss_function)
        fitted_data = fitted_data.dropna(subset=["fitted_dist"])
        data[f"fitted_dist_{dist_family.name}"] = fitted_data["fitted_dist"]
        data[f"mean_sq_err_{dist_family.name}"] = df_calc_err(
            fitted_data,
            "fitted_dist",
            question,
            framing,
        )

    # Sort the data based on the sort_by parameter
    if sort_by:
        if len(sort_by) == 1:
            top_n = data.nlargest(n, f"mean_sq_err_{sort_by[0].name}")
        elif len(sort_by) == 2:
            fam1, fam2 = sort_by
            data["mse_diff"] = data[f"mean_sq_err_{fam1.name}"] - data[f"mean_sq_err_{fam2.name}"]
            top_n = data.nlargest(n, "mse_diff")
        else:
            raise ValueError("sort_by must be None, a 1-tuple, or a 2-tuple of DistFamily objects")
    else:
        top_n = data.head(n)

    rows = (n + 4) // 5  # Round up to the nearest multiple of 5
    fig, axs = plt.subplots(rows, 5, figsize=(25, 5 * rows))
    if not sort_by:
        sort_by_human = "No sorting"
    else:
        sort_by_human = f"Top {n} CDFs by "
        if len(sort_by) == 1:
            sort_by_human += f"{sort_by[0].name} MSE"
        elif len(sort_by) == 2:
            sort_by_human += f"{fam1.name} minus {fam2.name} MSE"

    fig.suptitle(f"{question} {framing}: {sort_by_human}")

    for idx, (_, row) in enumerate(top_n.iterrows()):
        ax = axs[idx // 5, idx % 5] if rows > 1 else axs[idx]

        plot_individual_cdf(ax, row, question, framing, dist_families, colors)

        if sort_by:
            if len(sort_by) == 1:
                ax.set_title(f"MSE: {sort_by[0].name}={row[f'mean_sq_err_{sort_by[0].name}']:.2e}")
            else:
                fam1, fam2 = sort_by
                ax.set_title(
                    f"MSEs: {fam1.name}:{row[f'mean_sq_err_{fam1.name}']:.2e} vs {fam2.name}:{row[f'mean_sq_err_{fam2.name}']:.2e}"
                )

    # Hide any unused subplots
    for idx in range(n, rows * 5):
        ax = axs[idx // 5, idx % 5] if rows > 1 else axs[idx]
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def prev_fits_bias_hist():
    for question in [Question.HLMI, Question.FAOL]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{question.name}")

        for row, framing in enumerate([Framing.FY, Framing.FP]):
            data = merged(question)
            data = filter_rename(data, question, framing)

            cdf_data = data[framing.human_columns]

            if framing == Framing.FY:
                # Source data expresses probabilities as values between 0 and 100
                cdf_data = cdf_data / 100

            fitted_data = df_fitted_values(
                data,
                {"shape": "prev_shape", "scale": "prev_scale"},
                question,
                framing,
                DistFamily.GAMMA,
            )

            if framing == Framing.FY:
                col_unit = "Probability"
            elif framing == Framing.FP:
                col_unit = "Years"

            for i, col in enumerate(framing.human_columns):
                ax = axs[row, i]
                difference = fitted_data[f"{col}_fitted"] - cdf_data[col]

                if framing == Framing.FY:
                    p = 5 / 100
                    domain = (-p, p)
                elif framing == Framing.FP:
                    y = 75
                    domain = (-y, y)

                # Add colored backgrounds
                ax.axvspan(domain[0], 0, facecolor="lightyellow")
                ax.axvspan(0, domain[1], facecolor="lightgreen")

                ax.hist(difference, bins=80, range=domain, alpha=0.5)

                p10 = np.percentile(difference, 10)
                p90 = np.percentile(difference, 90)
                expectation = np.mean(difference)

                if framing == Framing.FY:
                    p10label = f"10th percentile ({p10:.1%})"
                    p90label = f"90th percentile ({p90:.1%})"
                    expectationlabel = f"Expectation ({expectation:.1%})"
                elif framing == Framing.FP:
                    p10label = f"10th percentile ({p10:.0f})"
                    p90label = f"90th percentile ({p90:.0f})"
                    expectationlabel = f"Expectation ({expectation:.0f})"

                ax.axvline(x=0, color="black", linestyle="--")
                ax.axvline(x=p10, color="orange", label=p10label)
                ax.axvline(x=p90, color="red", label=p90label)
                ax.axvline(x=expectation, color="purple", linestyle="-", label=expectationlabel)

                ax.set_xlabel(f"Fitted minus actual {col} ({col_unit})")

                ax.yaxis.set_visible(False)

                ax.set_title(f"{framing}, {col}")
                ax.set_xlim(domain)
                ax.set_ylim(0, 50)

                if framing == Framing.FY:
                    # format axis as percentage
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

                ax.legend()

        plt.tight_layout()
        plt.show()


def compare_configurations(configs: List[Config], question: Question, ax):
    """
    Compare different configurations of the same question, framing, and family.
    """
    for config in configs:
        plot_aggregated_cdf(config, question, ax)

    ax.set_title(question.value)


def plot_aggregated_cdf(config: Config, question: Question, ax):
    """
    To keep the code simpler, doesn't support using the previous gamma fits as a comparison.
    Only the values in the ``DistFamily`` enum are supported. See the ``vs_previous``
    function for that functionality.
    """
    data = pd.read_csv(PROJECT_ROOT / "data" / "combined-cleaned-personal-anon.csv")
    data = filter_rename(data, question, config.framing)
    data = df_fit(data, question, config.framing, config.family, config.loss_function)
    data = data.dropna(subset=["fitted_dist"])

    # Generate years array
    survey_year = 2023

    plot_to_year = 2125

    plot_to_year_rel = plot_to_year - survey_year
    rel_years = np.linspace(0, plot_to_year_rel, 1000)

    framing_short = "all" if config.framing is None else config.framing.name
    framing_str = f"frm={framing_short}"
    family_str = config.family.name
    match config.aggregation:
        case AggMethod.ARITH_MEAN_PROBS:
            aggregation_short = "Mean(p)"
        case AggMethod.MEDIAN_PROBS:
            aggregation_short = "Med(p)"
        case AggMethod.GEO_MEAN_ODDS:
            aggregation_short = "GeoMean(odd)"
        case AggMethod.GEO_MEAN_ODDS_WINSORIZED:
            aggregation_short = "GeoMean(Winz(odds))"
        case AggMethod.ARITH_MEAN_YEARS:
            aggregation_short = "Mean(y)"
        case _:
            raise ValueError(f"Invalid aggregation method: {config.aggregation}")

    aggregation_str = f"agg={aggregation_short}"

    loss_str = "loss=NA" if config.loss_function is None else f"loss={config.loss_function.name}"

    label = f"{framing_str}, {family_str}, {aggregation_str}, {loss_str}"

    if config.aggregation.vertical:
        evaluate_at = rel_years
    elif config.aggregation.horizontal:
        evaluate_at = np.linspace(0, 1, 1000)

    # Quantiles of the distribution of individual responses, along the aggregation axis (
    # horizontal or vertical). The area between these will be shaded.
    indiv_dists_probability_mass = 0.5
    each_tail_mass = (1 - indiv_dists_probability_mass) / 2
    indiv_dists_quantiles = [each_tail_mass, 1 - each_tail_mass]
    summary = summarize_dists(
        data,
        evaluate_at,
        dist_col="fitted_dist",
        method=config.aggregation,
        dists_quantiles=indiv_dists_quantiles,
    )

    shaded_area_str = (
        f"{indiv_dists_quantiles[0]*100:.3g}-{indiv_dists_quantiles[1]*100:.3g}th percentile"
    )
    if config.aggregation.vertical:
        shaded_area_str = f"{shaded_area_str}(p)"
    elif config.aggregation.horizontal:
        shaded_area_str = f"{shaded_area_str}(y)"

    label_fillbetween = f"{framing_str}, {family_str}, {shaded_area_str}, {loss_str}"
    if config.aggregation.vertical:
        (line,) = ax.plot(survey_year + rel_years, summary["aggregated"], label=label)
        ax.fill_between(
            survey_year + rel_years,
            summary[indiv_dists_quantiles[0]],
            summary[indiv_dists_quantiles[1]],
            alpha=0.2,
            color=line.get_color(),
            label=label_fillbetween,
        )
    elif config.aggregation.horizontal:
        (line,) = ax.plot(summary["aggregated"], np.linspace(0, 1, 1000), label=label)
        ax.fill_betweenx(
            np.linspace(0, 1, 1000),
            summary[indiv_dists_quantiles[0]],
            summary[indiv_dists_quantiles[1]],
            alpha=0.2,
            color=line.get_color(),
            label=label_fillbetween,
        )

    ax.set_xlabel("Years")
    ax.set_ylabel("Probability")
    ax.set_title(question.value)
    ax.legend()
    ax.set_xlim(survey_year, plot_to_year)
    ax.set_ylim(CDF_YLIM)

    return ax


def plot_random_cdfs(configs: List[Config], question: Question, n: int = 5):
    """
    Plot n randomly selected CDFs for a list of Config objects.

    Args:
    - configs: List of Config objects
    - question: The question to analyze
    - n: Number of CDFs to plot (default: 5)
    """
    data = merged(question)
    data = filter_rename(data, question, None)

    # Fit data for all configs
    fitted_data = {}
    for config in configs:
        fitted = df_fit(data, question, config.framing, config.family, config.loss_function)
        fitted_data[config] = fitted.dropna(subset=["fitted_dist"])

    # Randomly select n rows
    random_indices = np.random.choice(fitted_data[configs[0]].index, size=n, replace=False)

    # Set up the plot
    fig, axs = plt.subplots(n, 1, figsize=(10, 5 * n))
    fig.suptitle(f"Random CDFs for {question}")

    for i, idx in enumerate(random_indices):
        ax = axs[i] if n > 1 else axs

        # Plot source data as scatter
        row = data.loc[idx]
        x_values, y_values = get_x_p(row, question, config.framing)
        ax.scatter(x_values, y_values, color="black", alpha=0.6, label="Source Data")

        # Plot fitted distributions for each config
        for config in configs:
            try:
                fitted_dist = fitted_data[config].loc[idx, "fitted_dist"]
            except KeyError:
                print(f"No fitted distribution for {config}")

            x_range = np.linspace(0, max(x_values) * 1.2, 1000)
            y_fitted = fitted_dist.cdf(x_range)
            framing_short = "all" if config.framing is None else config.framing.name
            label = f"{framing_short}, {config.family.name}, {config.loss_function.name}"
            ax.plot(x_range, y_fitted, label=label)

        ax.set_xlabel("Years")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(fontsize="x-small")
        ax.set_xlim(-0.1, max(x_values) * 1.2)
        ax.set_ylim(CDF_YLIM)

    plt.tight_layout()
    plt.show()
