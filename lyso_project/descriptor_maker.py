import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import MACCSkeys


def fetch_fp(mol, radius, n_bits=2048, use_chirality=True, bit_info=False):
    """
    Calculates Morgan fingerprint for a molecule.

    Parameters
    ----------
    mol : RDKit molecule object
        The molecule for which the fingerprint is calculated.
    radius : int
        The radius of the Morgan fingerprint.
    n_bits : int, optional
        The number of bits in the fingerprint. Default is 2048.
    use_chirality : bool, optional
        If True, uses chirality information in the fingerprint. Default is True.
    bit_info : bool, optional
        If True, returns the bit info of the fingerprint as well. Default is False.

    Returns
    -------
    morgan_fingerprint_as_array : numpy.ndarray
        The desired Morgan fingerprint as a numpy array.
    bi : dict
        The bit info of the fingerprint. Only returned if bit_info is True.
    """
    if bit_info == True:
        bi = {}
        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=use_chirality, bitInfo=bi
        )
        morgan_fingerprint_as_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_as_array)
        return morgan_fingerprint_as_array, bi

    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, useChirality=use_chirality
    )
    morgan_fingerprint_as_array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_as_array)

    return morgan_fingerprint_as_array


def fetch_fp_from_smiles(
    smiles, radius, n_bits=2048, use_chirality=True, bit_info=False
):
    """
    Calculates Morgan fingerprint for a molecule, if SMILES is provided as input.

    Parameters
    ----------
    smiles : str
        SMILES string of a molecule.
    radius : int
        Radius of the Morgan fingerprint.
    n_bits : int, optional
        Number of bits in the fingerprint. Default is 2048.
    use_chirality : bool, optional
        If True, uses chirality information in the fingerprint. Default is True.
    bit_info : bool, optional
        If True, returns the bit info of the fingerprint as well. Default is False.

    Returns
    -------
    morgan_fingerprint_as_array : numpy.ndarray
        Numpy array of the desired Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)

    if bit_info == True:
        return fetch_fp(mol, radius, n_bits, use_chirality, bit_info)

    return fetch_fp(mol, radius, n_bits, use_chirality)


def fetch_fp_from_df(
    df_in,
    radius,
    target_col="smiles",
    n_bits=2048,
    as_df=False,
    use_chirality=True,
    bit_info=True,
):
    """
    Calculates Morgan fingerprint for molecules in the dataframe.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe, where the SMILES column is expected to be present.
    radius : int
        Radius of the Morgan fingerprint.
    target_col : str, optional
        Name of the column containing the SMILES strings. Default is "smiles".
    n_bits : int, optional
        Number of bits in the fingerprint. Default is 2048.
    as_df : bool, optional
        If True, returns a dataframe of the calculated fingerprints. Default is False.
    use_chirality : bool, optional
        If True, uses chirality information in the fingerprint. Default is True.
    bit_info : bool, optional
        If True, returns the bit info of the fingerprint as well. Default is True.

    Returns
    -------
    df_out : pandas.DataFrame or numpy.ndarray
        Dataframe with the Morgan fingerprints as columns or a numpy array of the fingerprints.

    Raises
    ------
    ValueError
        If the SMILES column is not present in the dataframe.
    """
    df_in = df_in.copy()
    df_out = df_in.copy()

    # Lower case all the column names of the dataframe
    df_in.columns = df_in.columns.str.lower()

    # Checking whether the SMILES column is present in the dataframe
    if target_col not in df_in.columns:
        raise ValueError("SMILES column not present in the dataframe")

    # Checking whether the Molecules column is present in the dataframe, if not present, adding it
    if "molecules" not in df_in.columns:
        df_in["molecules"] = df_in[target_col].apply(lambda x: Chem.MolFromSmiles(x))

    # if bit_info is true, calculate it with the Morgan fingerprints and return both
    if bit_info == True:
        bi_list = []
        fp_list = []
        for mol in df_in["molecules"]:
            fp, bi = fetch_fp(mol, radius, n_bits, use_chirality, bit_info=True)
            bi_list.append(bi)
            fp_list.append(fp)
        if as_df == True:
            df_out[f"{radius}_fp"] = fp_list
            return df_out, bi_list
        else:
            return np.stack(fp_list), bi_list

    # Calculating the Morgan fingerprints
    fp = df_in["molecules"].apply(lambda x: fetch_fp(x, radius, n_bits, use_chirality))

    if as_df == True:
        df_out[f"{radius}_fp"] = fp
        return df_out

    else:
        return np.stack(fp)


def fetch_avalon_fp(mol, n_bits=2048):
    """
    Calculates Avalon fingerprint for a molecule.

    Parameters
    ----------
    mol : RDKit molecule object
        The molecule for which the fingerprint is calculated.
    n_bits : int, optional
        The number of bits in the fingerprint. Default is 2048.

    Returns
    -------
    avalon_fingerprint_as_array : numpy.ndarray
        The desired Avalon fingerprint as a numpy array.
    """
    avalon_fingerprint = fpAvalon.GetAvalonFP(mol, nBits=n_bits)
    avalon_fingerprint_as_array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(avalon_fingerprint, avalon_fingerprint_as_array)

    return avalon_fingerprint_as_array


def fetch_avalon_fp_from_smiles(smiles, n_bits=2048):
    """
    Calculates Avalon fingerprint for a molecule, if SMILES is provided as input.

    Parameters
    ----------
    smiles : str
        SMILES string of a molecule.
    n_bits : int, optional
        Number of bits in the fingerprint. Default is 2048.

    Returns
    -------
    avalon_fingerprint_as_array : numpy.ndarray
        Numpy array of the desired Avalon fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    return fetch_avalon_fp(mol, n_bits)


def fetch_avalon_fp_from_df(df_in, target_col="smiles", n_bits=2048, as_df=False):
    """
    Calculates Avalon fingerprint for molecules in the dataframe.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe, where the SMILES column is expected to be present.
    target_col : str, optional
        Name of the column containing the SMILES strings. Default is "smiles".
    n_bits : int, optional
        Number of bits in the fingerprint. Default is 2048.
    as_df : bool, optional
        If True, returns a dataframe of the calculated fingerprints. Default is False.

    Returns
    -------
    df_out : pandas.DataFrame or numpy.ndarray
        Dataframe with the Avalon fingerprints as columns or a numpy array of the fingerprints.

    Raises
    ------
    ValueError
        If the SMILES column is not present in the dataframe.
    """
    df_in = df_in.copy()
    df_out = df_in.copy()

    # Lower case all the column names of the dataframe
    df_in.columns = df_in.columns.str.lower()

    # Checking whether the SMILES column is present in the dataframe
    if target_col not in df_in.columns:
        raise ValueError("SMILES column not present in the dataframe")

    # Checking whether the Molecules column is present in the dataframe, if not present, adding it
    if "molecules" not in df_in.columns:
        df_in["molecules"] = df_in[target_col].apply(lambda x: Chem.MolFromSmiles(x))

    # Calculating the Avalon fingerprints
    fp = df_in["molecules"].apply(lambda x: fetch_avalon_fp(x, n_bits))

    if as_df == True:
        df_out[f"{n_bits}_avalon_fp"] = fp
        return df_out

    else:
        return np.stack(fp)


def fetch_maccs_fingerprint(mol):
    """
    Calculates MACCS fingerprint for a molecule.

    Parameters
    ----------
    mol : RDKit molecule object
        The molecule for which the fingerprint is calculated.

    Returns
    -------
    maccs_fingerprint_as_array : numpy.ndarray
        The desired MACCS fingerprint as a numpy array.
    """
    maccs_fingerprint = MACCSkeys.GenMACCSKeys(mol)
    maccs_fingerprint_as_array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(maccs_fingerprint, maccs_fingerprint_as_array)

    return maccs_fingerprint_as_array


def fetch_maccs_fingerprint_from_smiles(smiles):
    """
    Calculates MACCS fingerprint for a molecule, if SMILES is provided as input.

    Parameters
    ----------
    smiles : str
        SMILES string of a molecule.

    Returns
    -------
    maccs_fingerprint_as_array : numpy.ndarray
        Numpy array of the desired MACCS fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    return fetch_maccs_fingerprint(mol)


def fetch_maccs_fingerprint_from_df(df_in, target_col="smiles", as_df=False):
    """
    Calculates MACCS fingerprint for molecules in the dataframe.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe, where the SMILES column is expected to be present.
    target_col : str, optional
        Name of the column containing the SMILES strings. Default is "smiles".
    as_df : bool, optional
        If True, returns a dataframe of the calculated fingerprints. Default is False.

    Returns
    -------
    df_out : pandas.DataFrame or numpy.ndarray
        Dataframe with the MACCS fingerprints as columns or a numpy array of the fingerprints.

    Raises
    ------
    ValueError
        If the SMILES column is not present in the dataframe.
    """
    df_in = df_in.copy()
    df_out = df_in.copy()

    # Lower case all the column names of the dataframe
    df_in.columns = df_in.columns.str.lower()

    # Checking whether the SMILES column is present in the dataframe
    if target_col not in df_in.columns:
        raise ValueError("SMILES column not present in the dataframe")

    # Checking whether the Molecules column is present in the dataframe, if not present, adding it
    if "molecules" not in df_in.columns:
        df_in["molecules"] = df_in[target_col].apply(lambda x: Chem.MolFromSmiles(x))

    # Calculating the MACCS fingerprints
    fp = df_in["molecules"].apply(lambda x: fetch_maccs_fingerprint(x))

    if as_df == True:
        df_out["maccs_fp"] = fp
        return df_out

    else:
        return np.stack(fp)


def rdkit_descriptor_calculator(
    df_in, add_logP=False, add_bpKa1=False, add_bpKa2=False, custom_descriptor_list=None
):
    """
    Calculates select RDKit descriptors for the molecules in the dataframe.

    Raises error if no SMILES column in present in the query dataframe.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe, where the SMILES column is expected to be present.
    add_logP : bool, optional
        If True, adds logP to the list of descriptors. Default is False.
    add_bpKa1 : bool, optional
        If True, adds bpKa1 to the list of descriptors. Default is False.
    add_bpKa2 : bool, optional
        If True, adds bpKa2 to the list of descriptors. Default is False.
    custom_descriptor_list : list, optional
        Custom list of descriptors provided by the user instead. Default is None.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of the calculated descriptors.

    Raises
    ------
    ValueError
        If the SMILES column is not present in the dataframe.
    """

    df_in = df_in.copy()

    # In past, I got NaNs with these descriptors, so I am not using them
    not_used_desc = [
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        "BCUT2D_MWHI",
        "BCUT2D_MWLOW",
        "BCUT2D_CHGHI",
        "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI",
        "BCUT2D_LOGPLOW",
        "BCUT2D_MRHI",
        "BCUT2D_MRLOW",
    ]

    # Appending 'MolLogP' because we already have calculated logP using ChemAxon Marvin, and 'Ipc' as its value is too high
    not_used_desc.extend(["MolLogP", "Ipc"])

    # used descriptors
    descriptors_list = [
        x for x in [x[0] for x in Chem.Descriptors.descList] if x not in not_used_desc
    ]

    # if a custom descriptor list is provided, then we use that instead of the default list
    if custom_descriptor_list is not None:
        descriptors_list = custom_descriptor_list

    # Create a descriptor calculator with select descriptors
    desc_calc = MolecularDescriptorCalculator(descriptors_list)

    # Lower case all the column names of the dataframe
    df_in.columns = df_in.columns.str.lower()

    # Checking whether the SMILES column is present in the dataframe
    if "smiles" not in df_in.columns:
        raise ValueError("SMILES column not present in the dataframe")

    # Checking whether the Molecules column is present in the dataframe, if not present, adding it
    if "molecules" not in df_in.columns:
        df_in["molecules"] = df_in["smiles"].apply(lambda x: Chem.MolFromSmiles(x))

    desc = []
    # Looping over the molecules
    for mol in df_in["molecules"]:
        desc.append(desc_calc.CalcDescriptors(mol))

    df_desc = pd.DataFrame(desc, columns=descriptors_list)

    # Adding logP and bpKa1 and bpKa2 if required
    if add_logP:
        df_desc["logP"] = df_in["logp"].values

    if add_bpKa1:
        df_desc["bpKa1"] = df_in["bpka1"].values

    if add_bpKa2:
        df_desc["bpKa2"] = df_in["bpka2"].values

    return df_desc


def cp_desc_isolator(df_in):
    """
    Isolates the CP features from the dataframe.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe.

    Returns
    -------
    cp_out : pandas.DataFrame
        Dataframe containing only the CP descriptors.
    """
    cp_out = df_in.loc[:, df_in.columns.str.contains("Median")].copy()

    return cp_out


def cp_desc_limiter(df_in):
    """
    First checks whether all columns are CP features or not. If not, raises error.
    Then caps the maximum limit to 25 and minimum limit to -25 for all the features.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe.

    Returns
    -------
    df_out : pandas.DataFrame
        Capped dataframe.

    Raises
    ------
    ValueError
        If not all columns are CP features.
    """

    df_in = df_in.copy()

    if df_in.columns.str.contains("Median").all() == False:
        raise ValueError("Not all columns are CP features")

    df_clipped = df_in.clip(-25, 25)
    return df_clipped


def minmax_scaling(df_in, as_df=False):
    """
    Scales the dataframe using MinMaxScaler.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe.
    as_df : bool, optional
        If True, returns a dataframe, else returns a numpy array. Default is False.

    Returns
    -------
    scaled_output : numpy.ndarray or pandas.DataFrame
        Scaled output as a numpy array or Pandas dataframe.
    scaler : sklearn.preprocessing.MinMaxScaler
        MinMaxScaler object.
    """

    df_in = df_in.copy()

    scaler = MinMaxScaler()
    scaled_output = scaler.fit_transform(df_in)

    if as_df == True:
        scaled_output = pd.DataFrame(scaled_output, columns=df_in.columns)

    return scaled_output, scaler


def date_extractor(df_in, extract_date=True, extract_year=True):
    """
    Uses Plate ID from the CP dataset and extract date and year from it.
    Adds a column "Plate_Date" and "Plate_Year" to the dataframe.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Query dataframe.
    extract_date : bool, optional
        If True, extracts date from the Plate ID. Default is True.
    extract_year : bool, optional
        If True, extracts year from the Plate ID. Default is True.

    Returns
    -------
    df_in : pandas.DataFrame
        Dataframe with date and year columns.
    """
    df_in = df_in.copy()

    splitted_plate = df_in["Plate"].str.rsplit("-", n=1, expand=True)
    if extract_date:
        df_in["Plate_Date"] = splitted_plate[1].copy()
    if extract_year:
        year = splitted_plate[1].str.slice(stop=2)
        df_in["Plate_Year"] = year.copy()

    return df_in
