// Lock-based parallel octree construction.
//
// Every thread inserts its own share of the particles, starting from the
// root. Child slots are the only shared state that is modified, and they are
// modified with atomic compare-and-swap:
//
//   empty slot            CAS  EMPTY -> leaf(i)
//   occupied leaf         CAS  leaf(q) -> LOCKED, build the replacement
//                         subtree privately, then publish it with one store
//   leaf at MAX_DEPTH     CAS  leaf(q) -> leaf(i) after setting next[i] = q
//
// A thread that reads LOCKED, or loses a CAS, re-reads the slot and tries
// again. The result is the same tree the sequential builder produces (only
// the node numbering and the order inside multi-particle leaves differ).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <omp.h>

#include "bh/stages.hpp"

namespace bh {
namespace {

// Marks a child slot whose subtree is being replaced. Only exists during the build.
constexpr std::int32_t LOCKED = std::numeric_limits<std::int32_t>::min();

// Node indices are handed out in blocks, so threads touch the shared counter
// once per kBlock nodes instead of once per node.
constexpr std::int64_t kBlock = 64;

// Atomic access to a child slot. The slots live in a plain int32 array that
// the other stages read without synchronization, so the compiler's atomic
// builtins are used here (same operations as std::atomic<int32_t>).
inline std::int32_t load(const std::int32_t* slot) { return __atomic_load_n(slot, __ATOMIC_ACQUIRE); }
inline void publish(std::int32_t* slot, std::int32_t v) { __atomic_store_n(slot, v, __ATOMIC_RELEASE); }
inline bool cas(std::int32_t* slot, std::int32_t expected, std::int32_t desired) {
    return __atomic_compare_exchange_n(slot, &expected, desired, false, __ATOMIC_ACQ_REL,
                                       __ATOMIC_ACQUIRE);
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

struct NodePool {
    std::atomic<std::int64_t> next{1};    // first index not yet handed out (0 is the root)
    std::int64_t capacity = 0;
    std::atomic<bool> overflow{false};    // set when the node arrays are too small
};

struct LocalBlock {
    std::int32_t cur = 0, end = 0;  // this thread's unused node indices
};

// Returns a fresh private node, or -1 when the pool is exhausted.
inline std::int32_t alloc_node(Octree& t, NodePool& pool, LocalBlock& b, real node_size) {
    if (b.cur == b.end) {
        const std::int64_t start = pool.next.fetch_add(kBlock, std::memory_order_relaxed);
        if (start + kBlock > pool.capacity) {
            pool.overflow.store(true, std::memory_order_relaxed);
            return -1;
        }
        b.cur = static_cast<std::int32_t>(start);
        b.end = static_cast<std::int32_t>(start + kBlock);
    }
    const std::int32_t id = b.cur++;
    std::int32_t* ch = &t.child[8 * static_cast<std::size_t>(id)];
    for (int k = 0; k < 8; ++k) ch[k] = Octree::EMPTY;
    t.size[id] = node_size;
    return id;
}

// Inserts particle i. Returns false if the node pool overflowed.
bool insert(const Particles& p, Octree& t, NodePool& pool, LocalBlock& blk, std::int32_t i) {
    const double x = p.x[i], y = p.y[i], z = p.z[i];
    double cx = t.cx, cy = t.cy, cz = t.cz, half = 0.5 * t.root_size;
    std::int32_t* const child = t.child.data();
    std::int32_t node = 0;
    int depth = 0;
    t.next[i] = -1;

    for (;;) {
        const int k = octant(x, y, z, cx, cy, cz);
        std::int32_t* const slot = child + 8 * static_cast<std::size_t>(node) + k;
        const std::int32_t c = load(slot);

        if (c == LOCKED) {
            if (pool.overflow.load(std::memory_order_relaxed)) return false;
            cpu_relax();
            continue;
        }
        if (Octree::is_node(c)) {
            enter_child(k, cx, cy, cz, half);
            node = c;
            ++depth;
            continue;
        }
        if (c == Octree::EMPTY) {
            if (cas(slot, Octree::EMPTY, Octree::make_leaf(i))) return true;
            continue;
        }

        // The octant holds a leaf whose first particle is q.
        const std::int32_t q = Octree::leaf_head(c);
        if (depth + 1 >= Octree::MAX_DEPTH) {
            // The child cell cannot be split: push i onto the front of the leaf's chain.
            t.next[i] = q;
            if (cas(slot, c, Octree::make_leaf(i))) return true;
            continue;
        }
        if (!cas(slot, c, LOCKED)) continue;

        // The slot is ours. Build the nodes that separate i and q privately,
        // then make them visible with a single store.
        double kx = cx, ky = cy, kz = cz, kh = half;
        enter_child(k, kx, ky, kz, kh);
        int d = depth + 1;  // depth of the node being filled
        const double qx = p.x[q], qy = p.y[q], qz = p.z[q];

        const std::int32_t first = alloc_node(t, pool, blk, static_cast<real>(2 * kh));
        if (first < 0) return false;
        std::int32_t cur = first;
        for (;;) {
            const int ki = octant(x, y, z, kx, ky, kz);
            const int kq = octant(qx, qy, qz, kx, ky, kz);
            std::int32_t* const cc = child + 8 * static_cast<std::size_t>(cur);
            if (ki != kq) {
                cc[ki] = Octree::make_leaf(i);
                cc[kq] = c;
                break;
            }
            if (d + 1 >= Octree::MAX_DEPTH) {
                t.next[i] = q;
                cc[ki] = Octree::make_leaf(i);
                break;
            }
            enter_child(ki, kx, ky, kz, kh);
            ++d;
            const std::int32_t deeper = alloc_node(t, pool, blk, static_cast<real>(2 * kh));
            if (deeper < 0) return false;
            cc[ki] = deeper;
            cur = deeper;
        }
        publish(slot, first);
        return true;
    }
}

}  // namespace

void build_tree_lock(const Particles& p, const BBox& box, Octree& t) {
    const long long n = static_cast<long long>(p.size());
    t.set_root(box);
    t.parent_first = false;  // threads allocate node blocks independently
    t.next.resize(p.size());

    // A tree over N distinct particles usually needs about N / 2 nodes; close
    // pairs need more. If the estimate is too small the build is repeated with
    // twice the capacity, which then stays for the following time steps.
    std::size_t cap = static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(omp_get_max_threads()) * kBlock + 64;
    if (cap < t.capacity()) cap = t.capacity();

    for (;;) {
        t.reserve_nodes(cap);
        NodePool pool;
        pool.capacity = static_cast<std::int64_t>(t.capacity());
        for (int k = 0; k < 8; ++k) t.child[k] = Octree::EMPTY;
        t.size[0] = static_cast<real>(t.root_size);

        long long blank = 0;
#pragma omp parallel reduction(+ : blank)
        {
            LocalBlock blk;
#pragma omp for schedule(static) nowait
            for (long long i = 0; i < n; ++i) {
                if (pool.overflow.load(std::memory_order_relaxed)) continue;
                insert(p, t, pool, blk, static_cast<std::int32_t>(i));
            }
            // Indices left over in this thread's block stay in the node range; blank them.
            for (std::int32_t id = blk.cur; id < blk.end; ++id) {
                std::int32_t* ch = &t.child[8 * static_cast<std::size_t>(id)];
                for (int k = 0; k < 8; ++k) ch[k] = Octree::EMPTY;
                t.size[id] = t.mass[id] = t.comx[id] = t.comy[id] = t.comz[id] = real(0);
            }
            blank += blk.end - blk.cur;
        }

        if (!pool.overflow.load()) {
            t.n_nodes = static_cast<std::int32_t>(pool.next.load());
            t.blank_nodes = static_cast<std::int32_t>(blank);
            return;
        }
        cap = 2 * t.capacity();
    }
}

}  // namespace bh
