import isocor
import pandas as pd
import re
import numpy as np
import pubchempy
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole


def _adjust_NPH_met_formula(
    met_formula: str,
    derivative_formula: str,
):

    # I'll add the whole formula correction spiel here
    multipliers = [int(s) for s in re.findall(r'\d+', derivative_formula)]
    if len(multipliers) < 4:
        multipliers.append(1)
    formula_multipliers = [int(s) for s in re.findall(r'\d+', met_formula)]
    cs = formula_multipliers[0] - multipliers[0]
    hs = formula_multipliers[1] - multipliers[1]
    # currently there are no formulas that contain nitrogen
    #ns = formula_multipliers[2] - multipliers[2]
    os = formula_multipliers[3] - multipliers[3]
    new_formula = f'C{cs}H{hs}O{os}'

    return new_formula


def find_NPH_derivative(
    formula: str,
    derivatives: pd.DataFrame,
):
    # for now I'm writing it for NPH derivatization and we can 
    # differentiate the number of NPH molecules added by the
    # number of nitrogen atoms
    # This is not a general this, only applicable here
    der_regex = re.compile('N([0-9]*)')
    num_of_N = int(der_regex.findall(formula)[0])

    if num_of_N == 3:
        derivative_formula = derivatives.loc[0, 'formula']
        new_formula = _adjust_NPH_met_formula(
            met_formula=formula,
            derivative_formula=derivative_formula,
            )
        return derivative_formula, new_formula
    elif num_of_N == 6:
        derivative_formula = derivatives.loc[1, 'formula']
        new_formula = _adjust_NPH_met_formula(
            met_formula=formula,
            derivative_formula=derivative_formula,
            )
        return derivative_formula, new_formula
    elif num_of_N == 9:
        derivative_formula = derivatives.loc[2, 'formula']
        new_formula = _adjust_NPH_met_formula(
            met_formula=formula,
            derivative_formula=derivative_formula,
            )
        return derivative_formula, new_formula
    else:
        print(formula)
        print('There is an issue')


def _adjust_AA_met_formula(
    met_formula: str,
    derivative_formula: str,
):
    # I'll add the whole formula correction spiel here
    multipliers = [int(s) for s in re.findall(r'\d+', derivative_formula)]
    if len(multipliers) < 4:
        multipliers.append(1)
    formula_multipliers = [int(s) for s in re.findall(r'\d+', met_formula)]
    cs = formula_multipliers[0] - multipliers[0]
    hs = formula_multipliers[1] - multipliers[1]
    ns = formula_multipliers[2] - multipliers[2]
    os = formula_multipliers[3] - multipliers[3]
    if ns > 1:
        new_formula = f'C{cs}H{hs}N{ns}O{os}'
    else:
        new_formula = f'C{cs}H{hs}NO{os}'

    return new_formula


def find_AA_derivative(
    formula: str,
    derivatives: pd.DataFrame,
    exception: bool=False,
):
    # Similar to the method for NPH just for AA derivatization
    # No multiple additions of derivatization agent so it's easier
    # This is not a general this, only applicable here

    if exception:
        derivative_formula = derivatives.loc[1, 'formula']
        new_formula = _adjust_AA_met_formula(
            met_formula=formula,
            derivative_formula=derivative_formula,
            )
    else:
        derivative_formula = derivatives.loc[0, 'formula']
        new_formula = _adjust_AA_met_formula(
            met_formula=formula,
            derivative_formula=derivative_formula,
            )
    return derivative_formula, new_formula


def correct_TASQ_table(
    dataframe: pd.DataFrame,
    derivatization_method: str,
    derivatives: pd.DataFrame,
    tracer: str,
    purity: float,
):

    # Yes this is a reference to the cooking show
    final_table = []

    # find the individual metabolites
    metabolites = list(dataframe[~dataframe.loc[:, 'Name'].str.contains(tracer)].loc[:, 'Name'])

    # iterate through them
    for metabolite in metabolites:
        exceptions = ['Lysine']
        if metabolite in exceptions:
            exception = True
        else:
            exception = False
        # get the unlabeled formula
        original_formula = dataframe[dataframe.loc[:, 'Name'] == metabolite].loc[:, 'Formula'].to_string(index=False)
        if derivatization_method == 'NPH':
            # find the NPH derivative formula
            derivative, formula = find_NPH_derivative(original_formula, derivatives)
        elif derivatization_method == 'AA':
            # find the AA derivative formula
            if exception:
                derivative, formula = find_AA_derivative(original_formula, derivatives, exception=True)
            else:
                derivative, formula = find_AA_derivative(original_formula, derivatives, exception=False)

        # find the ascending mass order based on their name
        isotopologues = dataframe[dataframe.loc[:, 'Name'].str.contains(metabolite)]
        isotopologue_names = list(isotopologues.loc[:, 'Name'])
        regex = re.compile(tracer + '([0-9]*)')
        order = []

        for iso in isotopologue_names:
            try:
                order.append(int(regex.findall(iso)[0]))
            except:
                order.append(0)

        # create metabolite specific low resolution corrector
        corrector_LR = isocor.mscorrectors.LowResMetaboliteCorrector(
            formula=formula, tracer=tracer,
            derivative_formula=derivative,
            tracer_purity=[1-purity, purity],
            data_isotopes=isocor.mscorrectors.LowResMetaboliteCorrector.DEFAULT_ISODATA,
            correct_NA_tracer=False
            )

        # pick out the samples to iterate through them
        samples = list(dataframe.columns[4:])
        for sample in samples:
            sample_specific_df = isotopologues.loc[:, ['Name', sample]]
            mdv = [sample_specific_df.iloc[order.index(mplus), 1] for mplus in range(len(order))]

            corrected_area, iso_fraction, res, m_enr = corrector_LR.correct(mdv)

            to_add = [sample, metabolite, formula, derivative, corrected_area, iso_fraction, m_enr]
            final_table.append(to_add)

    data_table = pd.DataFrame(
        np.array(final_table, dtype=object),
        columns=[
            'sample',
            'metabolite',
            'formula',
            'derivative',
            'corrected_area',
            'iso_fraction',
            'm_enr'
            ]
        )

    return data_table


def convert_table_for_viz(
    reduced_data: pd.DataFrame,
):
    # isolate conditions based on the sample name
    conditions = list(set(reduced_data.loc[:, 'condition']))

    # to collect all the results
    mdv_results = []
    # iterate through conditions
    for condition in conditions:
        conditional_df = reduced_data.loc[reduced_data['condition'] == condition, :]

        # per metabolite per condition we will require one mean and one stdev value per isotopologue
        # get metabolites
        metabolites = list(set(conditional_df.loc[:, 'metabolite']))
        # iterate through metabolites
        for metabolite in metabolites:
            metabolite_df = conditional_df.loc[conditional_df['metabolite'] == metabolite, :].reset_index()
            # get the mdvs
            values = [[] for _ in range(len(metabolite_df.loc[0, 'iso_fraction']))]

            for _, row in metabolite_df.iterrows():
                mdv = row.loc['iso_fraction']
                # now get all the values so we can calculate the stats per isotopologue
                for position, value in enumerate(mdv):
                    values[position].append(value)
            mean_mdv = [np.nanmean(iso) for iso in values]
            stedv_mdv = [np.nanstd(iso) for iso in values]

            for i, _ in enumerate(mean_mdv):
                # to collect the specific results
                mdv_metabolite_results = []
                mdv_metabolite_results.append(metabolite)
                mdv_metabolite_results.append('M+' + str(i))
                mdv_metabolite_results.append(condition)
                mdv_metabolite_results.append(mean_mdv[i])
                mdv_metabolite_results.append(stedv_mdv[i])

                # add it to the overall results collections
                mdv_results.append(mdv_metabolite_results)

    data_table = pd.DataFrame(
        np.array(mdv_results, dtype=object),
        columns=[
            'Metabolite',
            'Mass',
            'Condition',
            'Mean',
            'Stedv',
            ]
        )

    return data_table


def get_metabolite_smiles(
    data_table: pd.DataFrame,
) -> dict:
    # This one takes a few seconds because it needs to talk to the pubchem API
    # The empty additional list is for potential fragment info

    metabolites = list(set(data_table.loc[:, 'metabolite']))

    metabolite_info = {}
    for metabolite in metabolites:
        for compound in pubchempy.get_compounds(metabolite, 'name'):
            metabolite_info[metabolite] = [compound.canonical_smiles, []]

    return metabolite_info


def plot_mid(
    metabolite: str,
    metabolite_info: dict,
    data_table: pd.DataFrame,
    savefile_folder_path: str = None,
):

    selected_metabolite = metabolite
    ##############################
    selected_metabolite_info = metabolite_info[selected_metabolite]
    ##############################
    m = Chem.MolFromSmiles(selected_metabolite_info[0])
    substructure = Chem.MolFromSmarts('C')
    fragment_indices = [ind-1 for ind in selected_metabolite_info[1]]
    c_positions = [element for tupl in m.GetSubstructMatches(substructure) for element in tupl]
    selected_atoms = [c_positions[atom_index] for atom_index in fragment_indices]
    m.__sssAtoms = selected_atoms
    ##############################
    dfCopy = data_table.loc[data_table['Metabolite'] == selected_metabolite, :].copy()
    duplicates = 100 # increase this number to increase precision
    for index, row in dfCopy.iterrows():
        for times in range(duplicates):
            new_row = row.copy()
            new_row['Mean'] = np.random.normal(row['Mean'],row['Stedv']) 
            dfCopy = dfCopy.append(new_row, ignore_index=True)
    ##############################
    pal = sns.color_palette("Pastel2", len(list(set(dfCopy.loc[:, 'Mass']))))

    fig = plt.figure(figsize=(10,5))
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Condition',
        y='Mean',
        hue='Mass',
        data=dfCopy,
        palette=pal,
        ci='sd',
        ).set(title=selected_metabolite)
    plt.legend(loc='upper right')
    plt.ylim((0, 1.02))
    #plt.xticks(rotation = 90)
    plt.grid(axis='y', color='0.75')

    im = Chem.Draw.MolToImage(m, highlightAtoms=selected_atoms)
    ax = plt.axes([0.9, 0.67, 0.19, 0.19], frameon=True)
    ax.imshow(im)
    ax.axis('off')

    if savefile_folder_path:
        plt.savefig(savefile_folder_path + selected_metabolite + '.jpg', bbox_inches="tight")