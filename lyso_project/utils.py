import pandas as pd
import numpy as np
import toml
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold, cross_validate


class ModelRunner:
    """
    Makes model objects for each input type.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def model_input(self, X_train, Y_train, X_test=None, Y_test=None, scaled=None):
        """
        Sets the model's input.

        Parameters
        ----------
        X_train : array-like
            Training data.
        Y_train : array-like
            Training labels.
        X_test : array-like, optional
            Test data. Default is None.
        Y_test : array-like, optional
            Test labels. Default is None.
        scaled : bool, optional
            If True, data is scaled. Default is None.

        Returns
        -------
        None
        """
        self.X_train = X_train
        self.Y_train = Y_train

        if X_test is not None and Y_test is not None:
            self.X_test = X_test
            self.Y_test = Y_test

        if scaled == True:
            self.model_name_out = f"{self.model_name}_scaled_model"
        elif scaled == False:
            self.model_name_out = f"{self.model_name}_unscaled_model"
        else:
            self.model_name_out = f"{self.model_name}_model"

    def make_model(self, scale_pos_weight: float, booster=None, n_jobs=-1, folds_cv=5):
        """
        Makes model object, performs Cross Validation, and then trains on complete train set.
        Final evaluation is done on test set if it exists.

        Parameters
        ----------
        scale_pos_weight : float
            Weight for positive class.
        booster : str, optional
            Type of booster. Default is None.
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is -1.
        folds_cv : int, optional
            Number of folds for cross validation. Default is 5.

        Returns
        -------
        None
        """

        if booster is not None:
            self.model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                booster=booster,
                eval_metric="logloss",
                n_jobs=n_jobs,
            )
        else:
            self.model = XGBClassifier(
                scale_pos_weight=scale_pos_weight, eval_metric="logloss", n_jobs=n_jobs
            )

        self.folds_cv = folds_cv
        # Perform k-fold CV
        scoring = {
            "kappa": make_scorer(cohen_kappa_score),
            "balanced_accuracy": "balanced_accuracy",
        }
        skfold = StratifiedKFold(n_splits=self.folds_cv, shuffle=True)

        scores = cross_validate(
            self.model,
            self.X_train,
            self.Y_train,
            cv=skfold,
            scoring=scoring,
            return_train_score=True,
        )

        self.kappa_cv = scores["test_kappa"]
        self.mean_kappa_cv = np.mean(scores["test_kappa"])
        self.balanced_accuracy_cv = scores["test_balanced_accuracy"]
        self.mean_balanced_accuracy_cv = np.mean(scores["test_balanced_accuracy"])

        # Model training on entire data
        self.model.fit(self.X_train, self.Y_train)

        # Model prediction and evaluation on test data, if it exists
        if hasattr(self, "X_test") and hasattr(self, "Y_test"):
            self.Y_pred = self.model.predict(self.X_test)

            df_test = pd.DataFrame()
            df_test["Lyso_Class_Original"] = self.Y_test
            df_test["Lyso_Class_Predicted"] = self.Y_pred

            self.balanced_accuracy_test = balanced_accuracy_score(
                df_test["Lyso_Class_Original"], df_test["Lyso_Class_Predicted"]
            )
            self.cohen_kappa_test = cohen_kappa_score(
                df_test["Lyso_Class_Original"], df_test["Lyso_Class_Predicted"]
            )
            self.confusion_matrix = confusion_matrix(
                df_test["Lyso_Class_Original"], df_test["Lyso_Class_Predicted"]
            )

        # Round values up to 2 decimal places
        self.kappa_cv = np.round_(self.kappa_cv, decimals=2)
        self.mean_kappa_cv = np.round_(self.mean_kappa_cv, decimals=2)
        self.balanced_accuracy_cv = np.round_(self.balanced_accuracy_cv, decimals=2)
        self.mean_balanced_accuracy_cv = np.round_(
            self.mean_balanced_accuracy_cv, decimals=2
        )

        if hasattr(self, "balanced_accuracy_test") and hasattr(
            self, "cohen_kappa_test"
        ):
            self.balanced_accuracy_test = np.round_(
                self.balanced_accuracy_test, decimals=2
            )
            self.cohen_kappa_test = np.round_(self.cohen_kappa_test, decimals=2)

    def save_model(self, path=None):
        """
        Saves model object.
        """
        # save model in JSON
        if path:
            self.model.save_model(f"{path}/{self.model_name_out}.json")
        else:
            self.model.save_model(f"{self.model_name_out}.json")

    def write_txt_log(self, filename: str, path=None, append=True):
        """
        Writes model cross validation results and evaluation results to a text file with model name.
        """
        full_file_name = f"{path}/{filename}.txt" if path else f"{filename}.txt"
        stars = "\n ************************************ \n"
        model_name_print = f"Model: {self.model_name_out} \n"
        balanced_accuracy_cv_print = f"Balanced accuracies of all the folds: {self.balanced_accuracy_cv}. Standard deviation: {np.round_(np.std(self.balanced_accuracy_cv), decimals = 2)}. \n"
        mean_balanced_accuracy_cv_print = f"{self.folds_cv}-fold Cross-Validation Balanced Accuracy score: {self.mean_balanced_accuracy_cv}. \n"
        kappa_cv_print = f"Cohen's Kappa scores of all the folds: {self.kappa_cv}. Standard deviation: {np.round_(np.std(self.kappa_cv), decimals= 2)}. \n"
        mean_kappa_cv_print = f"{self.folds_cv}-fold Cross-Validation Cohen's Kappa score: {self.mean_kappa_cv}. \n"

        all_text = [
            stars,
            model_name_print,
            balanced_accuracy_cv_print,
            mean_balanced_accuracy_cv_print,
            kappa_cv_print,
            mean_kappa_cv_print,
        ]

        if (
            hasattr(self, "balanced_accuracy_test")
            and hasattr(self, "cohen_kappa_test")
            and hasattr(self, "confusion_matrix")
        ):
            balanced_accuracy_print = (
                f"Test-set Balanced Accuracy score: {self.balanced_accuracy_test}. \n"
            )
            kappa_print = f"Test-set Cohen's Kappa score: {self.cohen_kappa_test}. \n"
            confusion_matrix_print = (
                f"Test-set Confusion Matrix: \n {self.confusion_matrix} \n"
            )
            all_text.extend(
                [balanced_accuracy_print, kappa_print, confusion_matrix_print]
            )

        all_text.append(stars)

        if append:
            with open(full_file_name, "a") as file:
                file.writelines(all_text)
        else:
            with open(full_file_name, "w") as file:
                file.writelines(all_text)

    def write_toml_logs(self, path=None):
        """
        Prepare a TOML file of logs for each model.
        """
        toml_filename = (
            f"{path}/{self.model_name_out}_toml_logs.toml"
            if path
            else f"{self.model_name_out}_toml_logs.toml"
        )

        toml_string = f"""
        #Model logs in TOML-formatted data

        Model_Name = "{self.model_name_out}"

        Numbers_of_Folds = {self.folds_cv}

        [Balanced_Accuracy_CV_Metrics]
        Balanced_Accuracy_of_all_Folds = {self.balanced_accuracy_cv.tolist()}
        Mean_Balanced_Accuracy_CV = {self.mean_balanced_accuracy_cv}
        Standard_Deviation_Mean_Balanced_Accuracy_CV = {np.round_(np.std(self.balanced_accuracy_cv), decimals = 2)}

        [Cohens_Kappa_CV_Metrics]
        Cohens_Kappa_of_all_Folds = {self.kappa_cv.tolist()}
        Mean_Cohens_Kappa_CV = {self.mean_kappa_cv}
        Standard_Deviation_Mean_Cohens_Kappa_CV = {np.round_(np.std(self.kappa_cv), decimals = 2)}
        """

        if (
            hasattr(self, "balanced_accuracy_test")
            and hasattr(self, "cohen_kappa_test")
            and hasattr(self, "confusion_matrix")
        ):
            toml_string += f"""
            [Test_Set_Metrics]
            Balanced_Accuracy_Test = {self.balanced_accuracy_test}
            Cohens_Kappa_Test = {self.cohen_kappa_test}
            Confusion_Matrix_Test = {self.confusion_matrix.tolist()}
            """

        with open(toml_filename, "w") as file:
            file.write(toml_string)
