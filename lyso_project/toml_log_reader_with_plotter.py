import toml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def toml_reader_from_directory(path_in: str):
    """
    Reads all TOML model log files present in the given directory and stores all the necessary performance metrics in individual lists.

    Parameters
    ----------
    path_in : str
        Path to the directory containing the TOML model log files.

    Returns
    -------
    model_details : dict
        A dictionary containing the following keys:
        - model_names: list of model names
        - mean_ba_cv: list of mean balanced accuracies across all the folds, of all the models
        - sd_ba_cv: list of standard deviation of mean balanced accuracies across all the folds, of all the models
        - mean_kappa_cv: list of mean Cohen's kappa across all the folds, of all the models
        - sd_kappa_cv: list of standard deviation of mean Cohen's kappa across all the folds, of all the models
        - ba_test: list of balanced accuracies of the test set of all the models, if 'Test_Set_Metrics' exists in the TOML file, else None
        - kappa_test: list of Cohen's kappa of the test set of all the models, if 'Test_Set_Metrics' exists in the TOML file, else None
    """
    files_in_dir = Path(path_in).glob("*.toml")

    if not directory_path_verifier(path_in):
        return

    model_names = []
    mean_ba_cv = []
    sd_ba_cv = []
    mean_kappa_cv = []
    sd_kappa_cv = []
    ba_test = []
    kappa_test = []

    for file in files_in_dir:
        single_model_logs = toml.load(file)

        model_names.append(single_model_logs["Model_Name"])
        mean_ba_cv.append(
            single_model_logs["Balanced_Accuracy_CV_Metrics"][
                "Mean_Balanced_Accuracy_CV"
            ]
        )
        sd_ba_cv.append(
            single_model_logs["Balanced_Accuracy_CV_Metrics"][
                "Standard_Deviation_Mean_Balanced_Accuracy_CV"
            ]
        )
        mean_kappa_cv.append(
            single_model_logs["Cohens_Kappa_CV_Metrics"]["Mean_Cohens_Kappa_CV"]
        )
        sd_kappa_cv.append(
            single_model_logs["Cohens_Kappa_CV_Metrics"][
                "Standard_Deviation_Mean_Cohens_Kappa_CV"
            ]
        )

        # Check if 'Test_Set_Metrics' exists in the dictionary
        if "Test_Set_Metrics" in single_model_logs:
            ba_test.append(
                single_model_logs["Test_Set_Metrics"].get(
                    "Balanced_Accuracy_Test_Set", None
                )
            )
            kappa_test.append(
                single_model_logs["Test_Set_Metrics"].get("Cohens_Kappa_Test_Set", None)
            )
        else:
            ba_test.append(None)
            kappa_test.append(None)

    model_details = {}
    model_details["model_names"] = model_names
    model_details["mean_ba_cv"] = mean_ba_cv
    model_details["sd_ba_cv"] = sd_ba_cv
    model_details["mean_kappa_cv"] = mean_kappa_cv
    model_details["sd_kappa_cv"] = sd_kappa_cv
    model_details["ba_test"] = ba_test
    model_details["kappa_test"] = kappa_test

    return model_details


def directory_path_verifier(path_in: str):
    """
    Verifies if the given directory path is valid.

    Parameters
    ----------
    path_in : str
        Path to the directory.

    Returns
    -------
    bool
        True if the path is valid, False otherwise.
    """
    if Path(path_in).is_dir():
        return True
    else:
        print("Invalid directory or path. Please enter a valid input.")
        return False


def plotter_cv(model_details, file_name=None, save_fig=False, dpi=150):
    """
    Make bar plots for the balanced accuracy and Cohen's kappa of different models' cross validation results.

    Parameters
    ----------
    model_details : dict
        Dictionary of model details.
    file_name : str, optional
        File name for the plot. Path can be included in the file name. Default is None.
    save_fig : bool, optional
        If True, the plot is saved in the given file path, else the plot is displayed. Default is False.
    dpi : int, optional
        Resolution of the plot. Default is 150.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(dpi=dpi)
    x_axis = np.arange(len(model_details["model_names"]))

    color = "tab:red"
    ax1.set_xlabel("Models")
    ax1.set_ylabel("Balanced Accuracy", color=color)
    ax1.bar(
        x_axis - 0.2,
        model_details["mean_ba_cv"],
        0.4,
        color=color,
        yerr=model_details["sd_ba_cv"],
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1)

    for i in range(0, len(x_axis)):
        ax1.text(
            i - 0.2,
            0.1,
            f"{model_details['mean_ba_cv'][i]}\n \u00B1 \n{model_details['sd_ba_cv'][i]}",
            ha="center",
            color="white",
            fontsize=8,
        )

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("Cohen's Kappa", color=color)
    ax2.bar(
        x_axis + 0.2,
        model_details["mean_kappa_cv"],
        0.4,
        color=color,
        yerr=model_details["sd_kappa_cv"],
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1)

    for i in range(0, len(x_axis)):
        ax2.text(
            i + 0.2,
            0.1,
            f"{model_details['mean_kappa_cv'][i]}\n \u00B1 \n{model_details['sd_kappa_cv'][i]}",
            ha="center",
            color="white",
            fontsize=8,
        )

    plt.xticks(x_axis, model_details["model_names"])
    fig.autofmt_xdate(rotation=45)
    ax1.tick_params(axis="x", labelsize=8)

    plt.title("Cross Validation: Model Performances")

    if save_fig == True:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.show()


def plotter_test_set(model_details, file_name=None, save_fig=False, dpi=150):
    """
    Make bar plots for the balanced accuracy and Cohen's kappa of different models' test set results.

    Parameters
    ----------
    model_details : dict
        Dictionary of model details.
    file_name : str, optional
        File name for the plot. Path can be included in the file name. Default is None.
    save_fig : bool, optional
        If True, the plot is saved in the given file path, else the plot is displayed. Default is False.
    dpi : int, optional
        Resolution of the plot. Default is 150.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(dpi=dpi)
    x_axis = np.arange(len(model_details["model_names"]))

    color = "tab:red"
    ax1.set_xlabel("Models")
    ax1.set_ylabel("Balanced Accuracy", color=color)
    ax1.bar(x_axis - 0.2, model_details["ba_test"], 0.4, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    for i in range(0, len(x_axis)):
        ax1.text(
            i - 0.2,
            model_details["ba_test"][i] + 0.01,
            model_details["ba_test"][i],
            ha="center",
            color=color,
            fontsize=8,
        )

    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("Cohen's Kappa", color=color)
    ax2.bar(x_axis + 0.2, model_details["kappa_test"], 0.4, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    for i in range(0, len(x_axis)):
        ax2.text(
            i + 0.2,
            model_details["kappa_test"][i] + 0.01,
            model_details["kappa_test"][i],
            ha="center",
            color=color,
            fontsize=8,
        )

    ax2.set_ylim(0, 1)

    plt.xticks(x_axis, model_details["model_names"])
    fig.autofmt_xdate(rotation=45)
    ax1.tick_params(axis="x", labelsize=8)

    plt.title("Test Set: Model Performances")

    if save_fig == True:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.show()


def toml_reader_from_dict_plus_plotter(
    path_in: str,
    file_name=None,
    plot_cv_results=True,
    plot_test_set_results=False,
    dpi=150,
):
    """
    Read the results of the cross validation and test set results from the TOML files in the given directory, and plot the results.

    Parameters
    ----------
    path_in : str
        Path to the directory containing the TOML files.
    file_name : str, optional
        File name for the plot. Path can be included in the file name. If not given, the plot is not saved and is only displayed.
    plot_cv_results : bool, optional
        Boolean value to plot the cross validation results. If file_name is not given, plot is only displayed. Default is True.
    plot_test_set_results : bool, optional
        Boolean value to plot the test set results. If file_name is not given, plot is only displayed. Default is False.
    dpi : int, optional
        Resolution of the plot. Default is 150.

    Returns
    -------
    None
    """

    model_details = toml_reader_from_directory(path_in)

    if plot_cv_results == True:
        if file_name == None:
            plotter_cv(model_details, dpi=dpi)
        else:
            file_name_mod = f"{file_name}_cv_results.png"
            plotter_cv(model_details, file_name=file_name_mod, save_fig=True, dpi=dpi)

    if plot_test_set_results == True:
        if file_name == None:
            plotter_test_set(model_details, dpi=dpi)
        else:
            file_name_mod = f"{file_name}_test_set_results.png"
            plotter_test_set(
                model_details, file_name=file_name_mod, save_fig=True, dpi=dpi
            )
