An input directory has the following structure:


```
input
│
├── raw_input
│   ├── movie_1   ← size: N(net_x) × N(net_y) × N(filters) × N(frames)
│   ├── movie_2
│   ├── movie_3
│   ⋮ 
│
├── preprocessed_input
│   ├── res_128x128_filters_sf3o3s2  ← name of specific preprocessing pipeline
│   │   ├── metadata.yaml            ← metadata common to all preprocessed input movies
│   │   ├── movie_1                  ← size: N(net_x) × N(net_y) × N(filters) × N(frames)
│   │   ├── movie_2
│   │   ├── movie_3
│   │   ⋮ 
│   ⋮
│
├── raw_input_seqs    ← organizes raw input movies into stimulus sequences
│   ├── seq_1         ← an ordered set of stimuli
│   │   ├── movie_1   ← symlink to `raw_input/movie_1`
│   │   ├── movie_2
│   │   ⋮
│   ⋮
│
├── preprocessed_input_seqs   ← organizes raw input movies into stimulus sequences
│   ├── seq_1_res_128x128_filters_sf3o3s2
│   │   ├── metadata.yaml     ← same metadata as in `preprocessed_input`
│   │   ├── movie_1           ← symlink to `preprocessed_input/movie_1`
│   │   ├── movie_2
│   │   ⋮
│   ⋮
│
└── stimuli   ← stimuli sequences ready to be given to the network
    ├── seq_1_res_128x128_filters_sf3o3s2_50picks  
    │      ↖
    ⋮       Pseudorandom list of file paths pertaining to a given preprocessed 
            stimulus sequence. Each entry corresponds to a “session”. Paths are
            relative to `input_dir`.
``` 
