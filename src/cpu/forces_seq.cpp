#include <cstddef>

#include "bh/force_kernel.hpp"
#include "bh/stages.hpp"

namespace bh {

void forces_seq(Particles& p, const Octree& tree, const Params& prm) {
    for (std::size_t i = 0; i < p.size(); ++i) force_on_particle(p, tree, prm, i);
}

}  // namespace bh
