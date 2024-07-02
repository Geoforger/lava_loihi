import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

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
def parse_log_file(file_path, limit=100):
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
                "postfiltered_spikes": [],
                "timings": [],
            }
        elif line.startswith("Num prefiltered spikes:"):
            value = int(line.split(": ")[1])
            current_test["prefiltered_spikes"].append(value)
        elif line.startswith("Num postfiltered spikes:"):
            value = int(line.split(": ")[1])
            current_test["postfiltered_spikes"].append(value)
        elif line.endswith("ns"):
            value = int(line.replace("ns", ""))
            current_test["timings"].append(value)

    if current_test and len(data) < limit:
        data.append(current_test)

    # For each test, find the min, max and mean timings
    data = data[:limit]
    df_parsed_data = pd.DataFrame(data)
    df_parsed_data["Min Time (ns)"] = df_parsed_data["timings"].apply(min)
    df_parsed_data["Max Time (ns)"] = df_parsed_data["timings"].apply(max)
    df_parsed_data["Mean Time (ns)"] = df_parsed_data["timings"].apply(
        lambda x: sum(x) / len(x) if x else 0
    )
    df_parsed_data["Timings (ms)"] = df_parsed_data["timings"].apply(
        lambda x: [t//1e6 for t in x]
    )
    df_parsed_data["Mean Time (ms)"] = df_parsed_data["Mean Time (ns)"].apply(
        lambda x: x / 1e6
    )

    # df_parsed_data["Mean Prefiltered"] = df_parsed_data["prefiltered_spikes"].apply(
    #     lambda x: np.round(np.mean(x))
    # )
    # df_parsed_data["Mean Postfiltered"] = df_parsed_data["postfiltered_spikes"].apply(
    #     lambda x: np.round(np.mean(x))
    # )

    # First step of each test should be removed as it counts the sim setup time
    df_parsed_data = df_parsed_data.applymap(remove_first_element)

    return df_parsed_data


def add_mean_line(mean_value, label, color):
    plt.axhline(mean_value, linestyle="--", label=f"{label} Mean", color=color)


def remove_first_element(val):
    if isinstance(val, list):
        return val[1:]
    return val


def main():
    # Read in log file
    # NO LOAD
    filter_crop_path = "./data/cam_latency/filter_crop.log"
    filter_no_crop_path = "./data/cam_latency/filter_no_crop.log"
    no_filter_crop_path = "./data/cam_latency/no_filter_crop.log"
    no_filter_no_crop_path = "./data/cam_latency/no_filter_no_crop.log"

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
    plt.title("Camera Process Latency under no load")
    plt.tight_layout()
    plt.savefig("./data/plots/cam_latency_no_load.eps", dpi=600)
    # plt.show()
    plt.close()

    # Combine all into a single dataframe
    combined_dict = {
        "Condition": ["Filter Crop", "Filter No Crop", "No Filter Crop", "No Filter No Crop"],
        "Mean Latency (ms)": [
            filter_crop_frame["Mean Time (ms)"].mean(),
            filter_no_crop_frame["Mean Time (ms)"].mean(),
            no_filter_crop_frame["Mean Time (ms)"].mean(),
            no_filter_no_crop_frame["Mean Time (ms)"].mean(),
        ],
    }

    combined_frame = pd.DataFrame(data=combined_dict)
    print("No Load Data")
    print(combined_frame)

    # Test under load tests
    load_filter_no_crop_path = "./data/cam_latency/load_filter_no_crop.log"
    load_no_filter_no_crop_path = "./data/cam_latency/load_no_filter_no_crop.log"
    load_filter_crop_path = "./data/cam_latency/load_filter_crop.log"
    load_no_filter_crop_path = "./data/cam_latency/load_no_filter_crop.log"

    load_filter_no_crop_frame = parse_log_file(load_filter_no_crop_path)
    load_no_filter_no_crop_frame = parse_log_file(load_no_filter_no_crop_path)
    load_filter_crop_frame = parse_log_file(load_filter_crop_path)
    load_no_filter_crop_frame = parse_log_file(load_no_filter_crop_path)

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
    plt.title("Camera Process Latency under load")
    plt.tight_layout()
    plt.savefig("./data/plots/cam_latency_load.eps", dpi=600)
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
    }

    combined_load_frame = pd.DataFrame(data=combined_load_dict)
    print("Under Load Data")
    print(combined_load_frame)

    # for t in range(len(load_filter_no_crop_frame)):
    #     test = load_filter_no_crop_frame.iloc[t]
    # for t in range(len(filter_crop_frame)):
    #     test = filter_crop_frame.iloc[t]
    #     print(test["timings"].index(max(test["timings"])))

    #     sb.scatterplot(data=test, x="prefiltered_spikes", y="timings")
    # plt.title("Process latency against number of incoming spikes")
    # plt.show()

    # combined_frame = pd.DataFrame(data=combined_dict)


if __name__ == "__main__":
    main()
