This folder simulates the satellite's onboard camera storage — the raw frames waiting
to be triaged by the edge pipeline.

It's empty by default. Populate it either with:
- `python edge/generate_sample_frames.py` (synthetic placeholder images, for demoing
  the pipeline/dashboard before you have real data or trained models), or
- a handful of real images copied over from your prepared dataset.

`run_edge_pipeline.py` treats every `.jpg`/`.jpeg`/`.png` file in here as one captured
frame.
