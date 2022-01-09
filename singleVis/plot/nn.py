
import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns


def main():
    datasets = ["mnist", "fmnist", "cifar10"]
    selected_epochs = [1, 4, 10]
    # k_neighbors = [10, 15, 20]
    k_neighbors = [15]
    col = np.array(["dataset", "method", "type", "hue", "k", "period", "eval"])
    df = pd.DataFrame({}, columns=col)

    for k in k_neighbors: # k neighbors
        for i in range(3): # dataset
            dataset = datasets[i]
            data = np.array([])
            # load data from evaluation.json
            # DVI
            content_path = "/home/xianglin/projects/DVI_data/TemporalExp/resnet18_{}".format(dataset)
            for epoch_id in range(3):
                epoch  = selected_epochs[epoch_id]
                eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2_A.json")
                with open(eval_path, "r") as f:
                    eval = json.load(f)
                nn_train = round(eval["nn_train_{}".format(k)], 3)
                nn_test = round(eval["nn_test_{}".format(k)], 3)

                if len(data) == 0:
                    data = np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(k), "{}".format(str(epoch_id)), nn_train]])
                else:
                    data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(k), "{}".format(str(epoch_id)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-Test", "{}".format(k), "{}".format(str(epoch_id)), nn_test]])), axis=0)

            eval_path = "/home/xianglin/projects/DVI_data/TemporalExp/resnet18_{}/Model/SV_evaluation.json".format(dataset)
            with open(eval_path, "r") as f:
                    eval = json.load(f)
            for epoch_id  in range(3):
                epoch = selected_epochs[epoch_id]
                nn_train = round(eval[k]["nn_train"][epoch], 3)
                nn_test = round(eval[k]["nn_test"][epoch], 3)

                data = np.concatenate((data, np.array([[dataset, "TimeVis", "Train", "TimeVis-Train", "{}".format(k), "{}".format(str(epoch_id)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "TimeVis", "Test", "TimeVis-Test", "{}".format(k), "{}".format(str(epoch_id)), nn_test]])), axis=0)

            df_tmp = pd.DataFrame(data, columns=col)
            df = df.append(df_tmp, ignore_index=True)
            df[["period"]] = df[["period"]].astype(int)
            df[["k"]] = df[["k"]].astype(int)
            df[["eval"]] = df[["eval"]].astype(float)

    #%%
    df.to_excel("nn.xlsx")
    for k in k_neighbors:
        df_tmp = df[df["k"] == k]
        pal20c = sns.color_palette('tab20c', 20)
        # sns.palplot(pal20c)
        sns.set_theme(style="whitegrid", palette=pal20c)
        hue_dict = {
            "DVI-Train": pal20c[0],
            "TimeVis-Train": pal20c[16],

            "DVI-Test": pal20c[3],
            "TimeVis-Test": pal20c[19],
        }
        sns.palplot([hue_dict[i] for i in hue_dict.keys()])

        axes = {'labelsize': 9,
                'titlesize': 9,}
        mpl.rc('axes', **axes)
        mpl.rcParams['xtick.labelsize'] = 9

        hue_list = ["DVI-Train", "DVI-Test", "TimeVis-Train", "TimeVis-Test"]

        fg = sns.catplot(
            x="period",
            y="eval",
            hue="hue",
            hue_order=hue_list,
            # order = [1, 2, 3, 4, 5],
            # row="method",
            col="dataset",
            ci=0.001,
            height=2.5, #2.65,
            aspect=1.0,#3,
            data=df_tmp,
            kind="bar",
            palette=[hue_dict[i] for i in hue_list],
            legend=True
        )
        sns.move_legend(fg, "lower center", bbox_to_anchor=(.42, 0.92), ncol=4, title=None, frameon=False)
        mpl.pyplot.setp(fg._legend.get_texts(), fontsize='9')

        axs = fg.axes[0]
        max_ = df_tmp["eval"].max()
        # min_ = df["eval"].min()
        axs[0].set_ylim(0., max_*1.1)
        axs[0].set_title("MNIST")
        axs[1].set_title("FMNIST")
        axs[2].set_title("CIFAR-10")

        (fg.despine(bottom=False, right=False, left=False, top=False)
         .set_xticklabels(['Begin', 'Mid', 'End'])
         .set_axis_labels("", "NN Preserving")
         )
        # fg.fig.suptitle("NN preserving property")

        fg.savefig(
            "nn_{}.pdf".format(k),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )


if __name__ == "__main__":
    main()

