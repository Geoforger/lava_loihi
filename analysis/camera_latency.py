import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

params = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    # "font.size": 16,
    # "axes.labelsize": 16,
    # "axes.titlesize": 16,
    # "xtick.labelsize": 10,
    # "ytick.labelsize": 10,
}

plt.rcParams.update(params)
cm = 1 / 2.54
sb.set_style("white")

# Function to parse the log file and limit to 100 tests
def parse_log_file(file_path, limit=100, filter_zeros=False):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    current_test = None

    for line in lines:
        line = line.strip()
        if line == "Start of Test":
            if current_test:
                data.append(current_test)
            if len(data) >= limit:
                break
            current_test = {
                "prefiltered_spikes": [],
                "precropped_spikes": [],
                "final_spikes": [],
                "timings": [],
            }
        elif line.startswith("Num prefiltered spikes:"):
            value = int(line.split(": ")[1])
            current_test["prefiltered_spikes"].append(value)
        elif line.startswith("Num precropped spikes:"):
            value = int(line.split(": ")[1])
            current_test["precropped_spikes"].append(value)
        elif line.startswith("Num final spikes:"):
            value = int(line.split(": ")[1])
            current_test["final_spikes"].append(value)
        elif line.endswith("ns"):
            value = int(line.replace("ns", ""))
            current_test["timings"].append(value)

    if current_test and len(data) < limit:
        data.append(current_test)

    # For each test, find the min, max and mean timings
    data = data[:limit]
    df_parsed_data = pd.DataFrame(data)

    # First step of each test should be removed as it counts the sim setup time
    df_parsed_data = df_parsed_data.applymap(remove_first_element)

    # Filter out the zero spike values for the load tests
    if filter_zeros is True:
        df_parsed_data = df_parsed_data.apply(remove_zero_timings, axis=1)

    df_parsed_data = df_parsed_data.apply(remove_high_values, axis=1)

    # Find all column values from processed data
    df_parsed_data["Min Time (ns)"] = df_parsed_data["timings"].apply(min)
    df_parsed_data["Max Time (ns)"] = df_parsed_data["timings"].apply(max)
    df_parsed_data["Mean Time (ns)"] = df_parsed_data["timings"].apply(np.mean)
    df_parsed_data["Timings (ms)"] = df_parsed_data["timings"].apply(
        lambda x: [t/1e6 for t in x]
    )
    df_parsed_data["Mean Time (ms)"] = df_parsed_data["Mean Time (ns)"].apply(
        lambda x: x / 1e6
    )
    df_parsed_data["Max Time (ms)"] = df_parsed_data["Max Time (ns)"].apply(
        lambda x: x / 1e6
    )
    df_parsed_data["Min Time (ms)"] = df_parsed_data["Min Time (ns)"].apply(
        lambda x: x / 1e6
    )
    df_parsed_data["Latency Deviation (ns)"] = df_parsed_data["timings"].apply(
        lambda x: np.std(x)
    )
    df_parsed_data["Latency Deviation (ms)"] = df_parsed_data[
        "Latency Deviation (ns)"
    ].apply(lambda x: x / 1e6)

    return df_parsed_data


def add_mean_line(mean_value, label, color):
    plt.axhline(mean_value, linestyle="--", label=f"{label} Mean", color=color)


def remove_first_element(val):
    if isinstance(val, list):
        return val[1:]
    return val


def remove_high_values(test, threshold=3):
    timings = test["timings"]
    if not timings:
        return test

    # Calculate the threshold dynamically based on standard deviation
    mean = np.mean(timings)
    std_dev = np.std(timings)
    upper_bound = mean + threshold * std_dev

    # Find indices of values above the upper bound
    high_value_indexes = [
        index for index, value in enumerate(timings) if value > upper_bound
    ]

    # Remove elements at high_value_indexes from all lists in the test
    for key in test.keys():
        if isinstance(test[key], list):
            test[key] = [
                value
                for index, value in enumerate(test[key])
                if index not in high_value_indexes
            ]

    return test


def remove_zero_timings(test):
    timings = test["prefiltered_spikes"]
    if not timings:
        return test

    zero_indexes = [index for index, value in enumerate(timings) if value == 0]

    # Remove elements at zero_indexes from all lists in the test
    for key in test.keys():
        if isinstance(test[key], list):
            test[key] = [
                value
                for index, value in enumerate(test[key])
                if index not in zero_indexes
            ]

    return test


def model_linear(x, a, b):
    return a * x + b


def model_quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def model_logarithmic(x, a, b):
    return a * np.log(x) + b


def model_nlogn(x, a, b):
    return a * x * np.log(x) + b


def model_expo(x, a, b):
    return a * np.exp(b * x)


def main(davis=True):
    ###################################
    # NO LOAD
    ###################################
    # Read in log file
    if davis is not True:
        cam = "dvXplorer"
    else:
        cam = "DAVIS240"

    filter_crop_path = f"./data/cam_latency/{cam}/filter_crop.log"
    filter_no_crop_path = f"./data/cam_latency/{cam}/filter_no_crop.log"
    no_filter_crop_path = f"./data/cam_latency/{cam}/no_filter_crop.log"
    no_filter_no_crop_path = f"./data/cam_latency/{cam}/no_filter_no_crop.log"

    # Parse the log file and get the structured data
    filter_crop_frame = parse_log_file(filter_crop_path)
    filter_no_crop_frame = parse_log_file(filter_no_crop_path)
    no_filter_crop_frame = parse_log_file(no_filter_crop_path)
    no_filter_no_crop_frame = parse_log_file(no_filter_no_crop_path)

    # Scatter plot of prefiltered and filtered spikes with labels for legend
    colors = sb.color_palette("tab10", n_colors=4)

    _, ax = plt.subplots(figsize=(8, 4))

    # Plot each dataset and add a mean line with defined colors
    sb.scatterplot(
        data=filter_crop_frame,
        x=filter_crop_frame.index,
        y="Mean Time (ms)",
        label="Filter Crop",
        ax=ax,
        color=colors[0],
    )
    add_mean_line(filter_crop_frame["Mean Time (ms)"].mean(), "Filter Crop", colors[0])

    sb.scatterplot(
        data=filter_no_crop_frame,
        x=filter_no_crop_frame.index,
        y="Mean Time (ms)",
        label="Filter No Crop",
        ax=ax,
        color=colors[1],
    )
    add_mean_line(filter_no_crop_frame["Mean Time (ms)"].mean(), "Filter No Crop", colors[1])

    sb.scatterplot(
        data=no_filter_crop_frame,
        x=filter_crop_frame.index,
        y="Mean Time (ms)",
        label="No Filter Crop",
        ax=ax,
        color=colors[2],
    )
    add_mean_line(no_filter_crop_frame["Mean Time (ms)"].mean(), "No Filter Crop", colors[2])

    sb.scatterplot(
        data=no_filter_no_crop_frame,
        x=filter_crop_frame.index,
        y="Mean Time (ms)",
        label="No Filter No Crop",
        ax=ax,
        color=colors[3],
    )
    add_mean_line(no_filter_no_crop_frame["Mean Time (ms)"].mean(), "No Filter No Crop", colors[3])

    # Set labels and title
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.xlabel("Test Number")
    plt.ylabel("Mean Time (ms)")
    plt.title(f"{cam} Camera Process Latency under no load")
    plt.tight_layout()
    plt.savefig(f"./data/plots/{cam}_latency_no_load.eps", dpi=600)
    # plt.show()
    plt.close()

    # Combine all into a single dataframe
    combined_dict = {
        "Condition": [
            "Filter Crop",
            "Filter No Crop",
            "No Filter Crop",
            "No Filter No Crop",
        ],
        "Mean Latency (ms)": [
            filter_crop_frame["Mean Time (ms)"].mean(),
            filter_no_crop_frame["Mean Time (ms)"].mean(),
            no_filter_crop_frame["Mean Time (ms)"].mean(),
            no_filter_no_crop_frame["Mean Time (ms)"].mean(),
        ],
        "Max Latency (ms)": [
            filter_crop_frame["Max Time (ms)"].max(),
            filter_no_crop_frame["Max Time (ms)"].max(),
            no_filter_crop_frame["Max Time (ms)"].max(),
            no_filter_no_crop_frame["Max Time (ms)"].max(),
        ],
        "Min Latency (ms)": [
            filter_crop_frame["Min Time (ms)"].min(),
            filter_no_crop_frame["Min Time (ms)"].min(),
            no_filter_crop_frame["Min Time (ms)"].min(),
            no_filter_no_crop_frame["Min Time (ms)"].min(),
        ],
        "Latency Deviation (ms)": [
            filter_crop_frame["Latency Deviation (ms)"].mean(),
            filter_no_crop_frame["Latency Deviation (ms)"].mean(),
            no_filter_crop_frame["Latency Deviation (ms)"].mean(),
            no_filter_no_crop_frame["Latency Deviation (ms)"].mean(),
        ],
    }

    combined_frame = pd.DataFrame(data=combined_dict)
    print("No Load Data")
    print(combined_frame)

    ###################################
    # Test under load tests
    ###################################
    load_filter_no_crop_path = f"./data/cam_latency/{cam}/load_filter_no_crop.log"
    load_no_filter_no_crop_path = f"./data/cam_latency/{cam}/load_no_filter_no_crop.log"
    load_filter_crop_path = f"./data/cam_latency/{cam}/load_filter_crop.log"
    load_no_filter_crop_path = f"./data/cam_latency/{cam}/load_no_filter_crop.log"

    load_filter_no_crop_frame = parse_log_file(load_filter_no_crop_path, filter_zeros=True)
    load_no_filter_no_crop_frame = parse_log_file(
        load_no_filter_no_crop_path, filter_zeros=True
    )
    load_filter_crop_frame = parse_log_file(load_filter_crop_path, filter_zeros=True)
    load_no_filter_crop_frame = parse_log_file(
        load_no_filter_crop_path, filter_zeros=True
    )

    _, ax = plt.subplots(figsize=(8, 4))

    # Plot each dataset and add a mean line with defined colors
    sb.scatterplot(
        data=load_filter_crop_frame,
        x=load_filter_crop_frame.index,
        y="Mean Time (ms)",
        label="Filter Crop",
        ax=ax,
        color=colors[0],
    )
    add_mean_line(
        load_filter_crop_frame["Mean Time (ms)"].mean(), "Filter Crop", colors[0]
    )

    sb.scatterplot(
        data=load_filter_no_crop_frame,
        x=load_filter_no_crop_frame.index,
        y="Mean Time (ms)",
        label="Filter No Crop",
        ax=ax,
        color=colors[1],
    )
    add_mean_line(
        load_filter_no_crop_frame["Mean Time (ms)"].mean(), "Filter No Crop", colors[1]
    )

    sb.scatterplot(
        data=load_no_filter_crop_frame,
        x=load_filter_crop_frame.index,
        y="Mean Time (ms)",
        label="No Filter Crop",
        ax=ax,
        color=colors[2],
    )
    add_mean_line(
        load_no_filter_crop_frame["Mean Time (ms)"].mean(), "No Filter Crop", colors[2]
    )

    sb.scatterplot(
        data=load_no_filter_no_crop_frame,
        x=load_filter_crop_frame.index,
        y="Mean Time (ms)",
        label="No Filter No Crop",
        ax=ax,
        color=colors[3],
    )
    add_mean_line(
        load_no_filter_no_crop_frame["Mean Time (ms)"].mean(),
        "No Filter No Crop",
        colors[3],
    )

    # Set labels and title
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.xlabel("Test Number")
    plt.ylabel("Mean Time (ms)")
    plt.title(f"{cam} Camera Process Latency under load")
    plt.tight_layout()
    plt.savefig(f"./data/plots/{cam}_latency_load.eps", dpi=600)
    # plt.show()
    plt.close()

    # Combine all into a single dataframe
    combined_load_dict = {
        "Condition": [
            "Filter Crop",
            "Filter No Crop",
            "No Filter Crop",
            "No Filter No Crop",
        ],
        "Mean Latency (ms)": [
            load_filter_crop_frame["Mean Time (ms)"].mean(),
            load_filter_no_crop_frame["Mean Time (ms)"].mean(),
            load_no_filter_crop_frame["Mean Time (ms)"].mean(),
            load_no_filter_no_crop_frame["Mean Time (ms)"].mean(),
        ],
        "Max Latency (ms)": [
            load_filter_crop_frame["Max Time (ms)"].max(),
            load_filter_no_crop_frame["Max Time (ms)"].max(),
            load_no_filter_crop_frame["Max Time (ms)"].max(),
            load_no_filter_no_crop_frame["Max Time (ms)"].max(),
        ],
        "Min Latency (ms)": [
            load_filter_crop_frame["Min Time (ms)"].min(),
            load_filter_no_crop_frame["Min Time (ms)"].min(),
            load_no_filter_crop_frame["Min Time (ms)"].min(),
            load_no_filter_no_crop_frame["Min Time (ms)"].min(),
        ],
        "Latency Deviation (ms)": [
            load_filter_crop_frame["Latency Deviation (ms)"].mean(),
            load_filter_no_crop_frame["Latency Deviation (ms)"].mean(),
            load_no_filter_crop_frame["Latency Deviation (ms)"].mean(),
            load_no_filter_no_crop_frame["Latency Deviation (ms)"].mean(),
        ],
    }

    combined_load_frame = pd.DataFrame(data=combined_load_dict)
    print("Under Load Data")
    print(combined_load_frame)

    ###################################
    # Analyse the effects of load
    ###################################
    conditions = [
        load_filter_no_crop_frame,
        load_no_filter_no_crop_frame,
        load_filter_crop_frame,
        load_no_filter_crop_frame
    ]

    cond_names = [
        "Filter No Crop",
        "No Filter No Crop",
        "Filer Crop",
        "No Filter Crop"
    ]

    patches = []

    for idx, cond in enumerate(conditions):
        name = cond_names[idx]
        for t in range(len(cond)):
            test = cond.iloc[t]
            sb.scatterplot(
                data=test, x="prefiltered_spikes", y="Timings (ms)", color=colors[idx]
            )

        patches.append(mpatches.Patch(color=colors[idx], label=name))

    plt.legend(handles=patches)
    plt.title(f"{cam} Process latency against number of incoming spikes")
    plt.xlabel("Prefiltered Spikes")
    plt.ylabel("Mean Latency (ms)")
    plt.tight_layout()
    plt.savefig(f"./data/plots/{cam}_latency_load_incoming_spikes.png", dpi=600, bbox_inches="tight")
    # plt.show()
    plt.close()

    ###################################
    # Determine time complexity of algorithm
    ###################################
    time_complex_frame = pd.DataFrame([])
    for d in conditions:
        time_complex_frame = pd.concat([time_complex_frame, d])

    prefiltered_spikes = time_complex_frame["prefiltered_spikes"].apply(
        lambda x: np.mean(x) if isinstance(x, list) else 0
    ).to_numpy()
    timings = time_complex_frame["Mean Time (ms)"].to_numpy()

    popt_linear, _ = curve_fit(
        model_linear,
        prefiltered_spikes,
        timings,
    )
    popt_quadratic, _ = curve_fit(
        model_quadratic,
        prefiltered_spikes,
        timings,
    )
    popt_logarithmic, _ = curve_fit(
        model_logarithmic,
        prefiltered_spikes,
        timings,
    )
    popt_nlogn, _ = curve_fit(
        model_nlogn,
        prefiltered_spikes,
        timings,
    )

    predicted_linear = model_linear(prefiltered_spikes, *popt_linear)
    predicted_quadratic = model_quadratic(prefiltered_spikes, *popt_quadratic)
    predicted_logarithmic = model_logarithmic(prefiltered_spikes, *popt_logarithmic)
    predicted_nlogn = model_nlogn(prefiltered_spikes, *popt_nlogn)

    for idx, cond in enumerate(conditions):
        name = cond_names[idx]
        for t in range(len(cond)):
            test = cond.iloc[t]
            sb.scatterplot(
                    data=test, x="prefiltered_spikes", y="Timings (ms)", color=colors[idx]
                )

        patches.append(mpatches.Patch(color=colors[idx], label=name))

    plt.plot(prefiltered_spikes, predicted_linear, label='Linear Fit', color='blue')
    plt.plot(prefiltered_spikes, predicted_quadratic, label='Quadratic Fit', color='green')
    plt.plot(prefiltered_spikes, predicted_logarithmic, label='Logarithmic Fit', color='red')
    plt.plot(prefiltered_spikes, predicted_nlogn, label='n log(n) Fit', color='purple')
    plt.title(f"{cam} Process latency against number of incoming spikes")
    plt.xlabel("Prefiltered Spikes")
    plt.ylabel("Mean Latency (ms)")
    plt.tight_layout()
    # plt.show()
    plt.close()

    # Calculate R-squared and RMSE for each model
    r2_linear = r2_score(timings, predicted_linear)
    rmse_linear = np.sqrt(mean_squared_error(timings, predicted_linear))

    r2_quadratic = r2_score(timings, predicted_quadratic)
    rmse_quadratic = np.sqrt(mean_squared_error(timings, predicted_quadratic))

    r2_logarithmic = r2_score(timings, predicted_logarithmic)
    rmse_logarithmic = np.sqrt(mean_squared_error(timings, predicted_logarithmic))

    r2_nlogn = r2_score(timings, predicted_nlogn)
    rmse_nlogn = np.sqrt(mean_squared_error(timings, predicted_nlogn))

    # Print R-squared and RMSE values
    print(f"Linear Model: R-squared = {r2_linear}, RMSE = {rmse_linear}")
    print(f"Quadratic Model: R-squared = {r2_quadratic}, RMSE = {rmse_quadratic}")
    print(f"Logarithmic Model: R-squared = {r2_logarithmic}, RMSE = {rmse_logarithmic}")
    print(f"n log(n) Model: R-squared = {r2_nlogn}, RMSE = {rmse_nlogn}")

    ###################################
    # Bar plots of each test condition
    ###################################
    # Combine data frames for plotting
    combined_frame["Type"] = "No Load"
    combined_load_frame["Type"] = "Load"
    df_combined = pd.concat([combined_frame, combined_load_frame])

    # Plot
    plt.figure(figsize=(12, 8))

    # Bar plot for no load (hatched bars with transparency)
    sb.barplot(
        data=df_combined[df_combined["Type"] == "No Load"],
        x="Condition",
        y="Mean Latency (ms)",
        palette="pastel",
        hatch="/",
        edgecolor="white",
        alpha=0.5,
        ci=None,
        zorder=2,
    )

    # Bar plot for load (solid bars)
    sb.barplot(
        data=df_combined[df_combined["Type"] == "Load"],
        x="Condition",
        y="Mean Latency (ms)",
        palette="pastel",
        ci=None,
        zorder=1,
    )

    # Add error bars for no load
    for index, row in combined_frame.iterrows():
        plt.errorbar(
            x=index,
            y=row["Mean Latency (ms)"],
            yerr=row["Latency Deviation (ms)"],
            fmt="none",
            c="black",
            capsize=5,
        )

    # Add error bars for load
    for index, row in combined_load_frame.iterrows():
        plt.errorbar(
            x=index,
            y=row["Mean Latency (ms)"],
            yerr=row["Latency Deviation (ms)"],
            fmt="none",
            c="red",
            capsize=5,
        )

    # Set plot labels and title
    plt.xlabel("Condition")
    plt.ylabel("Latency (ms)")
    plt.title(f"{cam} Camera Latency Under Load and No Load")
    plt.xticks(rotation=45)

    # Create custom legend
    hatched_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="/", label="No Load"
    )
    solid_patch = mpatches.Patch(color="blue", alpha=0.5, label="Load")
    plt.legend(handles=[hatched_patch, solid_patch])
    plt.tight_layout()
    plt.savefig(f"./data/plots/{cam}_latency_comparison.png", dpi=600, bbox_inches="tight")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    options = [True, False]
    for o in options:
        main(davis=o)
