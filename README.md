# diffusion_limited_aggregation

Computes a grid-based DLA using particle swarm diffusion. The method uses numpy's vectorized operations for scalability so larger particle swarms generally perform better.

Particles are generated randomly along a circular radius that begins at a fixed distance (e.g. 100 cells away) but later grows in proportion to the cluster. Each diffusion step consists of a single random step (vertically or horizontally) in the grid. 

A particle is removed from the diffusion swarm when it comes in contact with a static neighbor (joining the cluster) or strays too far from the cluster. In both cases, the particle is replaced with a new random particle

Adjacency can be determined by either 4-way or 8-way nearest neighbors, depending on the "diagonals" flag (True includes diagonals, making 8 neighbors).

The following example was generated from the default parameters in the notebook.

![An example of a DLA cluster with ~37,000 particles](example.png)
