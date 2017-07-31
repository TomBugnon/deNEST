An input directory has the following structure:


```
input
│
├── raw_input
│   ├── movie_1   ← size: N(movie_frames)*N(movie_x)* N(movie_y)
│   ├── movie_2
│   ├── movie_3
│   ⋮
│
├── preprocessed_input                            ← all preprocessed input movies. Preprocessing is specific to a given network and typically includes downsampling, contrast normalization and filtering.
│   ├── res_128x128_contrastnorm_filters_sf3o3s2  ← name of specific preprocessing pipeline.
│   │   ├── metadata.yaml                         ← metadata common to all preprocessed input movies and defining the preprocessing pipeline.
│   │   ├── movie_1                               ← size: N(frames) x N(filters) x N(net_x) × N(net_y)
│   │   ├── movie_2
│   │   ├── movie_3
│   │   ⋮
│   ⋮
│
├── raw_input_sets    ← organizes all the raw input movies into subsets (eg: 2 movies out of three)
│   ├── set_1         ← an unordered set of stimuli
│   │   ├── movie_1   ← symlink to `raw_input/movie_1`
│   │   ├── movie_2
│   │   ⋮
│   ⋮
│
├── preprocessed_input_sets   ← organizes all the preprocessed input movies into subsets. The sets are the same as those in raw_input_sets, for each preprocessing pipeline.
│   ├── set_1_res_128x128_contrastnorm_filters_sf3o3s2
│   │   ├── metadata.yaml     ← same metadata as in `preprocessed_input`
│   │   ├── movie_1           ← symlink to `preprocessed_input/res_128x128_contrastnorm_filters_sf3o3s2/movie_1`
│   │   ├── movie_2
│   │   ⋮
│   ⋮
│
└── stimuli   ← stimuli sequences ready to be given to the network
    ├── seq_1_set_1_res_128x128_contrastnorm_filters_sf3o3s2
    │      ↖
    ⋮       Pseudorandom list of file paths pertaining to a given preprocessed
            stimulus set (subset of all stimuli). Each entry defines an ordered
            sequence of stimuli, all drawn from the same given stimulus set, and
            corresponds to a “session”. Paths are relative to `input_dir`.
```
