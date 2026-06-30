# README

## Environment

This project has been tested on **Ubuntu 22.04 LTS**.

## Dependencies

Please install the following Python packages before running the reconstruction pipeline:

- `ismrmrd`
- `gadgetron`
- `finufft`
- `numpy`
- `cufinufft` (required for GPU reconstruction)
- `cupy` (required for GPU reconstruction)
- `nibabel` (used for saving reconstructed images in `.nii.gz` format)

## Installation

After installing **Gadgetron**, copy the following files into the corresponding directories.

### Configuration file

```text
zte_config.xml
```

Copy to:

```text
/home/<your_username>/anaconda3/envs/gadgetron/share/gadgetron/config/
```

### Python reconstruction module

```text
recon_zte
```

Copy to:

```text
/home/<your_username>/anaconda3/envs/gadgetron/share/gadgetron/python/
```

## Usage

Activate the Gadgetron Conda environment:

```bash
conda activate gadgetron
```

To reconstruct an ISMRMRD dataset (e.g., `example.h5`), run:

```bash
gadgetron_ismrmrd_client -f example.h5 -C zte_config.xml
```

## Output

The reconstruction pipeline generates the following outputs:

- Low-resolution reconstruction images
- Motion registration results
- Final reconstructed image (`total.nii.gz`)

By default, all output files are saved to:

```text
/home/user/result_image/
```

**Important:** Please create this directory before running the reconstruction pipeline, or modify the output path in the source code to a directory of your choice.