// Lock-based parallel tree construction. Threads insert their particles from
// the root and change child slots only with compare-and-swap:
//
//   empty slot          EMPTY   -> leaf(i)
//   occupied leaf       leaf(q) -> LOCKED, build the replacement privately, publish it
//   leaf at MAX_DEPTH   leaf(q) -> leaf(i), after next[i] = q
//
// On LOCKED or a failed CAS the slot is read again.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <omp.h>

#include "bh/stages.hpp"

namespace bh {
namespace {

constexpr std::int32_t LOCKED = std::numeric_limits<std::int32_t>::min();
constexpr std::int64_t kBlock = 64;  // node indices a thread takes from the pool at once

// The slots live in a plain int32 array that later stages read directly.
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
    std::atomic<std::int64_t> next{1};  // node 0 is the root
    std::int64_t capacity = 0;
    std::atomic<bool> overflow{false};
};

struct LocalBlock {
    std::int32_t cur = 0, end = 0;
};

// Returns -1 when the pool is exhausted.
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
    for (int k = 0; k < 8; ++k) t.child[8 * static_cast<std::size_t>(id) + k] = Octree::EMPTY;
    t.size[id] = node_size;
    return id;
}

// Returns false if the node pool overflowed.
bool insert(const Particles& p, Octree& t, NodePool& pool, LocalBlock& blk, std::int32_t i) {
    const std::int32_t leaf_i = Octree::make_leaf(i);
    const std::uint64_t key = t.key[i] = t.morton_key(p.x[i], p.y[i], p.z[i]);
    std::int32_t* const child = t.child.data();
    std::int32_t node = 0;
    double size = t.root_size;
    int shift = Octree::ROOT_SHIFT;
    t.next[i] = -1;

    for (;;) {
        std::int32_t* const slot = child + 8 * static_cast<std::size_t>(node) + ((key >> shift) & 7);
        const std::int32_t c = load(slot);

        if (c == LOCKED) {
            if (pool.overflow.load(std::memory_order_relaxed)) return false;
            cpu_relax();
            continue;
        }
        if (Octree::is_node(c)) {
            node = c;
            size *= 0.5;
            shift -= 3;
            continue;
        }
        if (c == Octree::EMPTY) {
            if (cas(slot, c, leaf_i)) return true;
            continue;
        }
        const std::int32_t q = Octree::leaf_head(c);
        if (shift == 0) {
            t.next[i] = q;
            if (cas(slot, c, leaf_i)) return true;
            continue;
        }
        if (!cas(slot, c, LOCKED)) continue;

        // The slot is ours: build the nodes that separate i and q, then publish them.
        const std::uint64_t key_q = t.key[q];
        size *= 0.5;
        const std::int32_t first = alloc_node(t, pool, blk, static_cast<real>(size));
        if (first < 0) return false;
        std::int32_t cur = first;
        for (shift -= 3;; shift -= 3) {
            const int ki = (key >> shift) & 7, kq = (key_q >> shift) & 7;
            std::int32_t* const cc = child + 8 * static_cast<std::size_t>(cur);
            if (ki != kq) {
                cc[ki] = leaf_i;
                cc[kq] = c;
                break;
            }
            if (shift == 0) {
                t.next[i] = q;
                cc[ki] = leaf_i;
                break;
            }
            size *= 0.5;
            const std::int32_t deeper = alloc_node(t, pool, blk, static_cast<real>(size));
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
    t.parent_first = false;
    t.next.resize(p.size());
    t.key.resize(p.size());

    // About N / 2 nodes are typical. If the capacity is too small the build is
    // repeated with twice as much, which is then kept for later steps.
    std::size_t cap = p.size() + static_cast<std::size_t>(omp_get_max_threads()) * kBlock + 64;
    if (cap < t.capacity()) cap = t.capacity();

    for (;; cap = 2 * t.capacity()) {
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
            for (long long i = 0; i < n; ++i)
                if (!pool.overflow.load(std::memory_order_relaxed))
                    insert(p, t, pool, blk, static_cast<std::int32_t>(i));

            // Indices left in this thread's block become blank nodes.
            for (std::int32_t id = blk.cur; id < blk.end; ++id) {
                for (int k = 0; k < 8; ++k) t.child[8 * static_cast<std::size_t>(id) + k] = Octree::EMPTY;
                t.size[id] = t.mass[id] = t.comx[id] = t.comy[id] = t.comz[id] = real(0);
            }
            blank += blk.end - blk.cur;
        }

        if (!pool.overflow.load()) {
            t.n_nodes = static_cast<std::int32_t>(pool.next.load());
            t.blank_nodes = static_cast<std::int32_t>(blank);
            return;
        }
    }
}

}  // namespace bh
