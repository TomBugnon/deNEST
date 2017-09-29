# HT2017_2o

This network is an update from the original network described in Hill&Tononi2005
and originally transcribed in NEST by Pablo, Leonardo Barbosa and Keiko.

It is a model of the thalamocortical network with two orientation selectivities in Vp.

The modifications from HT2005 are as follows:
- 1-  Make all the masks inclusive (12 -> 12.1, 7->7.1 etc)
- 2- Scale all the diffuse (mask > 4 units) connections of which the pool layer is Vs so that the masks of analogous connections have the same physical extent in Vp and Vs.
