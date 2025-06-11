"""Script to select tomogram slices to label and set-up min and max z-values for annotation and export slices as pngs in ChimeraX."""

from typing import List

state_dict = {}


def open_next_tomogram(session):
    from chimerax.markers import MarkerSet  # type: ignore
    from chimerax.core.colors import Color  # type: ignore
    from chimerax.map import Volume  # type: ignore

    global state_dict

    tomo_name = state_dict["tomograms"][state_dict["cur_idx"]]
    session.logger.info(f"Opening {tomo_name}")
    try:
        models, _ = session.open_command.open_data(
            str(state_dict["source"] / tomo_name)
        )
        # Create marker sets
        zlimit_markers = MarkerSet(session, name="zlimits")
        zlimit_markers.set_color(Color((1, 0, 0)))
        slice_markers = MarkerSet(session, name="slices")
        slice_markers.set_color(Color((0, 1, 0)))
        session.models.add(models)
        session.models.add([zlimit_markers, slice_markers])
        state_dict["zlim_markers"] = zlimit_markers
        state_dict["slice_markers"] = slice_markers
        # Change to plane view
        for model in models:
            if isinstance(model, Volume):
                pass
                # x, y, z = model.data.size
                # model.set_display_style("image")
                # model.new_region(ijk_min=(0, 0, z // 2), ijk_max=(x - 1, y - 1, z // 2))

    except Exception as e:
        session.logger.error(f"Error opening tomogram {tomo_name}: {e}")


def save_to_csv(session):
    import csv

    global state_dict

    csv_df = []
    try:
        with open(state_dict["csv"]) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                csv_df.append(row)
    except Exception as e:
        session.logger.error(
            f"Couldn't read information from {state_dict['csv']}. {e}."
        )
        return
    # Get data from state_dict
    z_limits = sorted([round(a.coord[2]) for a in state_dict["zlim_markers"].atoms])
    z_limits[1] += 1  # Adjust for end exclusion
    slices = sorted([round(a.coord[2]) for a in state_dict["slice_markers"].atoms])
    csv_data = {
        "tomo_name": state_dict["tomograms"][state_dict["cur_idx"]],
        "z_min": z_limits[0],
        "z_max": z_limits[1],
    }
    for i in range(len(slices)):
        csv_data[f"slice_{i}"] = slices[i]
    # Add/replace entry in previous .csv
    tomo_names = [row["tomo_name"] for row in csv_df]
    if csv_data["tomo_name"] in tomo_names:  # Replace
        idx = tomo_names.index(csv_data["tomo_name"])
        csv_df[idx] = csv_data
    else:  # Add
        csv_df.append(csv_data)

    # Write new .csv back to disk
    try:
        with open(state_dict["csv"], "w+") as csvfile:
            fieldnames = list(csv_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in csv_df:
                writer.writerow(item)
        session.logger.info(
            f"Saved {csv_data['tomo_name']} data to {state_dict['csv']}."
        )
    except Exception as e:
        session.logger.error(f"Couldn't save slices for {csv_data['tomo_name']}. {e}.")


def set_tomogram_slices(
    session,
    src_dir: str,
    sample: str,
    num_slices: int = 5,
    tomograms: List[str] = None,
    csv_dir: str = "",
):
    import os
    from pathlib import Path

    global state_dict

    # Check for directories
    src_dir = Path(src_dir).resolve()
    if not src_dir.is_dir() or not (src_dir / sample).is_dir():
        session.logger.warning(
            f"Invalid source (tomogram) directory {src_dir} with sample {sample}."
        )
    csv_path = Path(csv_dir).resolve() / f"{sample}.csv"
    src_path = src_dir / sample
    os.makedirs(csv_path.parent, exist_ok=True)

    # Save state
    state_dict = {
        "source": src_path,
        "csv": csv_path,
        "sample": sample,
        "num_slices": num_slices,
        "tomograms": tomograms or src_path.glob("*"),
        "cur_idx": 0,
        "z_lims": [],
        "slices": [],
    }
    # Preview the number of valid tomogram files
    session.logger.info(
        f"Starting annotations for {len(tomograms)} tomograms in sample {sample}."
    )
    # Start looping through all tomogram files
    open_next_tomogram(session)


def next_tomogram(session):
    from chimerax.std_commands.close import close  # type: ignore

    global state_dict

    # Check that labelling has started
    if not state_dict:
        session.logger.error(
            "Please start labelling tomograms first using 'start slice labels'."
        )
        return

    # Check if markers are placed
    if state_dict["zlim_markers"] and len(state_dict["zlim_markers"].atoms) != 2:
        session.logger.error(
            f"Please place two markers for z-limits. Currently there are {len(state_dict['zlim_markers'].atoms)} markers."
        )
        return
    if (
        state_dict["slice_markers"]
        and len(state_dict["slice_markers"].atoms) != state_dict["num_slices"]
    ):
        session.logger.error(
            f"Please place {state_dict['num_slices']} markers for slices. Currently there are {len(state_dict['slice_markers'].atoms)} markers."
        )
        return

    save_to_csv(session)
    # Close tomogram
    close(session)

    # Open next tomogram if available
    state_dict["cur_idx"] += 1
    if state_dict["cur_idx"] < len(state_dict["tomograms"]):
        open_next_tomogram(session)


def register_commands(logger):
    from chimerax.core.commands import (  # type: ignore
        CmdDesc,
        register,
        ListOf,
        OpenFolderNameArg,
        SaveFolderNameArg,
        StringArg,
        IntArg,
    )

    slices_desc = CmdDesc(
        required=[("src_dir", OpenFolderNameArg), ("sample", StringArg)],
        keyword=[
            ("num_slices", IntArg),
            ("tomograms", ListOf(StringArg)),
            ("csv_dir", SaveFolderNameArg),
        ],
        synopsis="Start setting the z-limits and slice numbers to label for a tomogram sample.",
    )
    next_desc = CmdDesc(
        required=[],
        keyword=[],
        synopsis="Finish processing current tomogram, add entry to .csv, and open the next tomogram to label.",
    )
    register("start selection", slices_desc, set_tomogram_slices, logger=logger)
    register("next", next_desc, next_tomogram, logger=logger)


register_commands(session.logger)  # type: ignore
