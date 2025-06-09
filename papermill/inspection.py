"""Deduce parameters of a notebook from the parameters cell."""

from pathlib import Path

import click

from .iorw import get_pretty_path, load_notebook_node, local_file_io_cwd
from .log import logger
from .parameterize import add_builtin_parameters, parameterize_path
from .translators import papermill_translators
from .utils import any_tagged_cell, find_first_tagged_cell_index, nb_kernel_name, nb_language, find_all_tagged_cell_indices, any_tagged_cells


def _open_notebook(notebook_path, parameters):
    path_parameters = add_builtin_parameters(parameters)
    input_path = parameterize_path(notebook_path, path_parameters)
    logger.info(f"Input Notebook:  {get_pretty_path(input_path)}")

    with local_file_io_cwd():
        return load_notebook_node(input_path)


def _infer_parameters(nb, name=None, language=None, parameter_tags=None):
    """Infer the notebook parameters from cells with specified tags.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        Notebook
    name : str, optional
        Kernel name override
    language : str, optional  
        Language override
    parameter_tags : list of str, optional
        Tags to search for parameter cells. Defaults to ['parameters']

    Returns
    -------
    List[Parameter]
       List of parameters (name, inferred_type_name, default, help)
    """
    if parameter_tags is None:
        parameter_tags = ['parameters']
        
    params = []

    # Find all parameter cells
    parameter_cell_indices = find_all_tagged_cell_indices(nb, parameter_tags)
    if not parameter_cell_indices:
        return params

    kernel_name = nb_kernel_name(nb, name)
    language = nb_language(nb, language)

    translator = papermill_translators.find_translator(kernel_name, language)
    
    # Collect parameters from all parameter cells
    seen_params = {}
    for idx in parameter_cell_indices:
        parameter_cell = nb.cells[idx]
        try:
            cell_params = translator.inspect(parameter_cell)
            for param in cell_params:
                if param.name in seen_params:
                    logger.warning(f"Parameter '{param.name}' defined in multiple cells. Using latest definition from cell {idx}.")
                seen_params[param.name] = param
                
        except NotImplementedError:
            logger.warning(f"Translator for '{language}' language does not support parameter introspection.")

    return list(seen_params.values())


def display_notebook_help(ctx, notebook_path, parameters, parameter_tags=None):
    """Display help on notebook parameters.

    Parameters
    ----------
    ctx : click.Context
        Click context
    notebook_path : str
        Path to the notebook to be inspected
    parameters : dict
        Parameters to pass to the notebook
    parameter_tags : list of str, optional
        Tags to search for parameter cells. Defaults to ['parameters']
    """
    if parameter_tags is None:
        parameter_tags = ['parameters']
        
    nb = _open_notebook(notebook_path, parameters)
    click.echo(ctx.command.get_usage(ctx))
    pretty_path = get_pretty_path(notebook_path)
    click.echo(f"\nParameters inferred for notebook '{pretty_path}':")

    if not any_tagged_cells(nb, parameter_tags):
        click.echo(f"\n  No cell tagged with any of {parameter_tags}")
        return 1

    params = _infer_parameters(nb, parameter_tags=parameter_tags)
    if params:
        for param in params:
            p = param._asdict()
            type_repr = p["inferred_type_name"]
            if type_repr == "None":
                type_repr = "Unknown type"

            definition = f"  {p['name']}: {type_repr} (default {p['default']})"
            if len(definition) > 30:
                if len(p["help"]):
                    param_help = f"{definition}\n{34 * ' '}{p['help']}"
                else:
                    param_help = definition
            else:
                param_help = f"{definition:<34}{p['help']}"
            click.echo(param_help)
    else:
        click.echo(
            "\n  Can't infer anything about this notebook's parameters. "
            "It may not have any parameters defined."
        )

    return 0


def inspect_notebook(notebook_path, parameters=None, parameter_tags=None):
    """Return the inferred notebook parameters.

    Parameters
    ----------
    notebook_path : str or Path
        Path to notebook
    parameters : dict, optional
        Arbitrary keyword arguments to pass to the notebook parameters
    parameter_tags : list of str, optional
        Tags to search for parameter cells. Defaults to ['parameters']

    Returns
    -------
    Dict[str, Parameter]
       Mapping of (parameter name, {name, inferred_type_name, default, help})
    """
    if isinstance(notebook_path, Path):
        notebook_path = str(notebook_path)

    nb = _open_notebook(notebook_path, parameters)

    params = _infer_parameters(nb, parameter_tags=parameter_tags)
    return {p.name: p._asdict() for p in params}
