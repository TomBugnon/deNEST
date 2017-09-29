# Hill and Tononi 2005 network

These parameters describe the network used by Pablo, Keiko and Leonardo in 2016/2017.

This network reproduces the behavior described in Hill&Tononi2005 when the scaling of the kernels is performed using the formula:

```
extent_units = grid_units * layer_extent / (layer_size - 1)
```

However this formula is incorrect in nest and should rather be :

```
extent_units = grid_units * layer_extent / (layer_size - 1)
```

When using the correct formula, the kernels and masks are slightly smaller which leads to the non inclusion of a significant number of units due to edge effects (~10000000 connections overall, vs ~12000000 with the incorrect formula). In that case the network activity simply dies in Vp_L23 and Vs.

The corrected version of the scaling therefore requires the masks to be 'inclusive' rather than 'exclusive' to lead to a correct network behavior. 
