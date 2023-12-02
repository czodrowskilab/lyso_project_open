def initial_filter(df_in, remove_DCM=True):
    """
    Apply initial filtering to the input dataframe.

    This function subsets the data where the compound concentration is 10uM, removes toxic compounds,
    entries with no structure, and compounds based on purity flag. It also asks users whether to remove
    DCM compounds or not, and arranges compounds in descending order based on 'Activity' column.

    Parameters
    ----------
    df_in : pandas.DataFrame
        The input dataframe to be processed.
    remove_DCM : bool, optional
        If True, removes DCM compounds. Default is True.

    Returns
    -------
    pandas.DataFrame
        The processed dataframe.

    """
    # Make a deep copy
    df = df_in.copy()

    # Let us subset the data where the compound conc is 10 uM. We also exclude toxic compounds and the ones with no structure
    df = df[
        # do only use measurements from the plates with the standard concentrations
        (df["Plate"].str.contains("-10.00-"))
        & (~df["Toxic"])  # exclude toxic compounds
        & (df["Smiles"].str.len() > 1)  # exclude No-structures ("*")
        & (~df["Pure_Flag"].str.contains("Fail"))
        & (~df["Pure_Flag"].str.contains("Warn"))
    ]

    df = df.reset_index(drop=True)

    if remove_DCM == True:
        df = df.query('not Plate.str.contains("DCM21")')
        df = df.reset_index(drop=True)

    # To keep the most active measurements for each compounds
    df = df.sort_values(
        "Activity", ascending=False
    )  # sort the data set by Induction in descending order (most active at the top)
    df = df.drop_duplicates(
        subset="Compound_Id"
    )  # keep only the most active measurement from each compound
    df = df.reset_index(drop=True)

    return df


def preprocessor_lyso_project(df_in):
    """
    Preprocess the input dataframe for the lyso project.

    This function adds a new column indicating whether a compound is lysosomotropic or not, removes all rows where bpKa1,
    bpKa2 and logP are NaN, checks if any logP values are NaN, removes apKa1 and apKa2 columns, and removes any
    bpKa1 NaN values.

    Parameters
    ----------
    df_in : pandas.DataFrame
        The input dataframe to be processed.

    Returns
    -------
    pandas.DataFrame or str
        The processed dataframe, or an error message if any logP values are NaN.

    """
    # Making a deep copy
    df = df_in.copy()

    # Adding a new col with the class
    df.loc[df["Lyso_Score"] < 75, "Lyso_Class"] = 0  # 0 = non-lysosomotropic
    df.loc[df["Lyso_Score"] >= 75, "Lyso_Class"] = 1  # 1 = lysosomotropic

    # Removing all rows where all bpKa1, bpKa2 and logP are NaN
    index_names = df[
        (df["bpKa1"].isnull()) & (df["bpKa2"].isnull()) & (df["logP"].isnull())
    ].index
    df.drop(index_names, inplace=True)
    df = df.reset_index(drop=True)

    # checking for nans in logP
    if df["logP"].isnull().sum():
        return f"Error! NaN value of logP present. Kindly check the data."

    # Removing apKa1 and apKa2 columns
    df = df.drop(["apKa1", "apKa2"], axis=1)

    # Removing NaN values of bpKa1
    df = df.dropna(subset=["bpKa1"])
    df = df.reset_index(drop=True)

    return df
