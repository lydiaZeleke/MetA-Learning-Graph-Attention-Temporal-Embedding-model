import pandas as pd
import matplotlib.pyplot as plt

def plot_anomalies_by_timeseries(
    enc_num_col='enc_num',
    label_col='label',
    time_step=0.1
):
    test_csv_path = f"datasets/CUSTOM/test.csv"
    label_csv_path = f"datasets/CUSTOM/label_data_test.csv"

    df_test = pd.read_csv(test_csv_path)
    df_labels = pd.read_csv(label_csv_path)
    assert len(df_test) == len(df_labels), "Mismatch in rows."
    df_test[label_col] = df_labels.iloc[:, 0].values

    grouped = df_test.groupby(enc_num_col)
    unique_groups = list(grouped.groups.keys())

    fig, axes = plt.subplots(nrows=len(unique_groups), ncols=1,
                             figsize=(8, 4 * len(unique_groups)),
                             sharex=False)

    if len(unique_groups) == 1:
        axes = [axes]

    for ax, (enc_val, group_df) in zip(axes, grouped):
        time_values = [i * time_step for i in range(len(group_df))]
        labels = group_df[label_col].values

        # ✅ Print the number of anomalies for each task
        anomaly_count = (labels == 1).sum()
        print(f"Enc_num {enc_val} — Anomaly count: {anomaly_count} — Task Length: {len(labels)}")

        ax.plot(time_values, labels, marker='o', linestyle='-', label=f'Enc_num={enc_val}')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(loc='best', fontsize=12)

    fig.text(0.04, 0.5, '', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()

plot_anomalies_by_timeseries()


# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_anomalies_by_timeseries(
#     enc_num_col='enc_num',
#     label_col='label',
#     time_step=0.1
# ):
#     """
#     Reads test data and label data from separate CSV files, combines them,
#     then plots binary anomaly labels by time for each unique 'enc_num'.
#     This version:
#       - Removes individual x‐axis labels (replacing them with a legend on each subplot).
#       - Has a single shared y‐axis label for all subplots.
#     """
    
#     test_csv_path = f"datasets/CUSTOM/test.csv"
#     label_csv_path = f"datasets/CUSTOM/label_data_test.csv"


#     df_test = pd.read_csv(test_csv_path)
#     df_labels = pd.read_csv(label_csv_path)
#     assert len(df_test) == len(df_labels), "Mismatch in rows."
#     df_test[label_col] = df_labels.iloc[:, 0].values

#     grouped = df_test.groupby(enc_num_col)
#     unique_groups = list(grouped.groups.keys())

#     fig, axes = plt.subplots(nrows=len(unique_groups), ncols=1,
#                              figsize=(8, 4 * len(unique_groups)),
#                              sharex=False)

#     if len(unique_groups) == 1:
#         # If there's only one group, axes won't be a list
#         axes = [axes]

#     for ax, (enc_val, group_df) in zip(axes, grouped):
#         time_values = [i * time_step for i in range(len(group_df))]
#         labels = group_df[label_col].values

#         # Plot and give the line a label so the legend can show the x-axis info
#         ax.plot(time_values, labels, marker='o', linestyle='-', label=f'Enc_num={enc_val}')
        
#         # ax.set_title(f"Enc_num={enc_val}")
        
#         # Remove individual x- and y-labels
#         ax.set_xlabel('')
#         ax.set_ylabel('')
        
#         # Fix the y-limits so anomalies stand out
#         ax.set_ylim(-0.1, 1.1)
        
#         # Show legend for each subplot (with the "Time" info)
#         ax.legend(loc='best', fontsize=12)


#     # Single y-axis label for entire figure
#     # fig.text(0.04, 0.5, 'Label (0=normal, 1=anomaly)', va='center', rotation='vertical')
#     fig.text(0.04, 0.5, '', va='center', rotation='vertical')

    
#     plt.tight_layout()
#     plt.show()

# plot_anomalies_by_timeseries()


