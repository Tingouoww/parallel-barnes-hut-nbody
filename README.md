# Barnes-Hut N 體模擬之樹建構與受力計算平行化

NYCU Parallel Programming（2026 Fall）期末專題。以 C++ 實作三維重力 Barnes-Hut N 體模擬的循序、OpenMP 與 CUDA 三種後端，比較樹建構與受力計算的平行化方法。

這份文件說明：怎麼編譯執行、輸出怎麼看、程式怎麼分層，以及每位組員要把自己的部分接在哪裡。

## 快速開始

需要 g++（支援 C++17 與 OpenMP）與 make。

```bash
make          # 產生 ./bh（單精度）
make test     # 跑驗證測試，約 1 分鐘
```

```bash
./bh --dist plummer --n 100000 --steps 10
```

Windows 上沒有 g++ 的話，可以用 Docker（在 PowerShell、repo 根目錄下）：

```powershell
docker run --rm -v "${PWD}:/work" -w /work gcc:12 make test
```

其他編譯選項：

| 指令 | 作用 |
|---|---|
| `make PRECISION=double` | 倍精度版本，產生 `./bh_double` |
| `make ARCH=native` | 加上 `-march=native`（各節點 CPU 相同時才用） |
| `make CUDA=1` | 一併編譯 `src/cuda/*.cu`，啟用 `--backend cuda`（這條路徑還沒有人測過） |
| `make clean` | 清除編譯產物 |

## 輸出怎麼看

```
$ ./bh --dist plummer --n 20000 --steps 2 --validate-every 1 --check-tree
bh: dist=plummer n=20000 seed=42 steps=2 theta=0.5 eps=0.01 dt=0.01 precision=float backend=seq build=seq force=seq threads=1
step    0  bbox 0.00002  build 0.00224  com 0.00057  force 0.18639  integrate 0.00004  total 0.18926 s  nodes 9682  inter/particle 1258.9
           error vs direct sum (1000 samples): median 1.425e-03  p99 9.741e-03  max 2.334e-02  rms 2.659e-03
           energy -0.14625199 (K 0.148900, U -0.295152)  drift +0.000e+00
           tree check passed
step    1  bbox 0.00003  build 0.00085  com 0.00033  force 0.16952  integrate 0.00006  total 0.17080 s  nodes 9727  inter/particle 1258.7
           error vs direct sum (1000 samples): median 1.449e-03  p99 9.577e-03  max 2.699e-02  rms 2.763e-03
           energy -0.14625224 (K 0.148899, U -0.295151)  drift -1.685e-06
           tree check passed
summary over 2 steps (median seconds per step, share of total)
  bbox       0.000027    0.0%
  build      0.001548    0.9%
  com        0.000451    0.3%
  force      0.177952   98.8%
  integrate  0.000053    0.0%
  total      0.180032
validation passed
```

- `step` 那一行是五個階段各自的秒數、八元樹的內部節點數（`nodes`），以及平均每個粒子的交互作用次數（`inter/particle`）。
- `error vs direct sum` 是與 O(N²) 直接加總相比的加速度相對誤差（只有加 `--validate-every` 才會出現）。
- `energy` 是系統總能量，`drift` 是相對於第一次驗證的漂移。
- `summary` 是各階段每步時間的中位數與佔比，`--warmup W` 可以排除前 W 步。

## 命令列選項

`./bh --help` 會列出全部選項。

| 選項 | 說明 | 預設 |
|---|---|---|
| `--dist` | 粒子分布：`uniform`（立方體內均勻分布）、`plummer`（Plummer 球）、`collision`（兩個 Plummer 球對撞）、`degenerate`（測試用，含重合粒子） | `plummer` |
| `--n`、`--seed` | 粒子數 N、亂數種子。相同的參數在任何平台都產生相同的粒子 | 10000、42 |
| `--steps`、`--warmup` | 時間步數；前幾步不列入 summary | 10、0 |
| `--theta` | 近似參數 θ | 0.5 |
| `--dt`、`--eps` | 時間步長 Δt、軟化長度 ε | 0.01、0.01 |
| `--backend` | `seq`（循序）、`omp`（OpenMP）、`cuda` | `seq` |
| `--build` | `omp` 後端的建樹方法：`seq`、`lock`（以鎖插入） | `seq` |
| `--force` | `omp` 後端的受力計算工作分配策略：目前只有 `seq` | `seq` |
| `--threads` | `omp` 後端的執行緒數 | OpenMP 預設 |
| `--validate-every K` | 每 K 步驗證一次（不計時） | 0（關閉） |
| `--samples M` | 每次驗證抽樣的粒子數 | 1000 |
| `--energy-max-n N` | N 不超過此值才計算總能量（位能是 O(N²)） | 20000 |
| `--check-tree` | 驗證時一併檢查樹的結構 | 關閉 |
| `--max-err X`、`--max-drift X` | p99 相對誤差或能量漂移超過 X 時，以狀態碼 2 結束（給測試腳本用） | 不檢查 |
| `--csv FILE` | 每個時間步附加一列到 FILE | 不輸出 |
| `--quiet` | 只印 summary | 關閉 |

單位：G = 1、總質量 = 1、Plummer 球的尺度半徑 = 1。

CSV 的欄位：`dist,n,seed,theta,eps,dt,precision,backend,build,force,threads,step,bbox_s,build_s,com_s,force_s,integrate_s,total_s,nodes,interactions,err_median,err_p99,err_rms,energy,energy_drift`。沒有驗證的時間步，最後五欄留空。

## 架構

模擬驅動器（`src/main.cpp`）在每個時間步依序呼叫後端的五個階段，並分別計時。循序、OpenMP、CUDA 三種後端實作同一個介面（`include/bh/backend.hpp`）。

| 階段 | `Backend` 的函式 | 輸出名稱 | 目前的實作 |
|---|---|---|---|
| 邊界框計算 | `compute_bbox()` | `bbox` | 循序、OpenMP |
| 樹建構 | `build_tree()` | `build` | 循序、以鎖插入 |
| 質心計算 | `compute_com()` | `com` | 循序 |
| 受力計算 | `compute_forces()` | `force` | 循序 |
| 積分 | `integrate()` | `integrate` | 循序、OpenMP |

驗證在受力計算與積分之間進行，不列入計時。

```
include/bh/
  types.hpp         real、Params、Particles、BBox
  octree.hpp        八元樹與 Morton 碼
  backend.hpp       Backend 介面
  stages.hpp        CPU 各階段的函式宣告
  force_kernel.hpp  單一粒子的樹走訪，CPU 的受力計算都呼叫它
  generator.hpp     工作負載產生器
  validate.hpp      驗證模組
src/
  main.cpp          模擬驅動器：時間步迴圈、計時、CSV
  generator.cpp
  validate.cpp
  cpu/
    cpu_backend.cpp   seq 與 omp 後端，內有策略表
    regular_stages.cpp  邊界框計算、積分
    tree_seq.cpp      循序建樹、質心計算
    tree_lock.cpp     以鎖插入建樹
    forces_seq.cpp    循序受力計算
  cuda/             CUDA 後端（尚未實作）
tests/
  run_tests.sh      驗證測試（make test）
  stress_build.sh   平行建樹的壓力測試，約 4 分鐘
```

## 資料結構

### 粒子（`Particles`）

Structure-of-arrays：`x y z vx vy vz ax ay az m` 各是一個長度 N 的陣列，型別是 `real`（預設 `float`）。另有 `cost[i]`，是粒子 i 在上一次受力計算的交互作用次數，供 costzones 與負載不均程度 λ 使用。

### 八元樹（`Octree`）

只儲存內部節點，節點 0 是根節點，全部以陣列表示（沒有指標）。

每個節點有 8 個子節點欄位 `child[8 * node + k]`，值有三種：

| 值 | 意義 |
|---|---|
| `-1`（`EMPTY`） | 這個子空間是空的 |
| `>= 0` | 內部節點的索引 |
| `<= -2` | 葉節點，`Octree::leaf_head(c)` 是其中粒子的索引 |

其他節點陣列：`size`（節點邊長 s）、`mass`、`comx`、`comy`、`comz`（總質量與質心，由質心計算階段填入）。

走訪一個節點的寫法：

```cpp
for (int k = 0; k < 8; ++k) {
    const std::int32_t c = tree.child[8 * node + k];
    if (c == Octree::EMPTY) continue;
    if (Octree::is_node(c)) {
        // c 是內部節點：tree.mass[c]、tree.comx[c]、tree.size[c] ...
    } else {
        // 葉節點：通常只有一個粒子，最大深度的葉節點可能有多個
        for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = tree.next[q]) {
            // 粒子 q
        }
    }
}
```

細分規則（與提案 3.2.1 節相同，所有建樹方法都要遵守，建出的樹才會一致）：

- 根節點是包住所有粒子的立方體。每個粒子的座標量化成每維 21 位元的整數，交錯成 63 位元的 Morton 碼，存在 `tree.key[i]`（由 `Octree::morton_key()` 計算）。
- 粒子屬於哪個子節點由 Morton 碼決定：深度 d 的節點取 `(key >> (60 - 3 * d)) & 7`，其中 bit 0、1、2 分別代表 x、y、z 的上半部。
- 節點只剩一個粒子就停止細分；空的子節點不建立；只有一個子節點的路徑不壓縮。
- 最大深度是 21。達到最大深度的葉節點可以容納多個粒子（也就是 Morton 碼相同的粒子），以 `tree.next[]` 串接，`-1` 結束。

節點陣列的長度是容量，有效節點是 `[0, tree.num_nodes())`。平行建樹讓每個執行緒一次領 64 個節點索引，沒用完的會留在範圍內成為空白節點（沒有子節點、從根節點走不到），數量記在 `tree.blank_nodes`。從根節點往下走訪不會遇到它們；直接掃整個節點陣列的程式要略過。

## 如何加入自己的部分

### 受力計算的工作分配策略（莊明儒）

每個粒子的受力計算是 `force_on_particle(p, tree, prm, i)`（`include/bh/force_kernel.hpp`），它寫入 `p.ax/ay/az[i]` 與 `p.cost[i]`，不同粒子之間互不影響。各策略都呼叫這個函式，只改變粒子怎麼分給執行緒，這樣比較的就只是工作分配。

以 `dynamic` 為例，三個步驟：

1. 在 `include/bh/stages.hpp` 宣告：

   ```cpp
   void forces_dynamic(Particles& p, const Octree& tree, const Params& prm);
   ```

2. 新增 `src/cpu/forces_omp.cpp`（Makefile 會自動編譯 `src/cpu/` 下的所有 `.cpp`）：

   ```cpp
   #include "bh/force_kernel.hpp"
   #include "bh/stages.hpp"

   namespace bh {

   void forces_dynamic(Particles& p, const Octree& tree, const Params& prm) {
       const long long n = static_cast<long long>(p.size());
   #pragma omp parallel for schedule(dynamic, 64)
       for (long long i = 0; i < n; ++i) force_on_particle(p, tree, prm, i);
   }

   }  // namespace bh
   ```

3. 在 `src/cpu/cpu_backend.cpp` 的 `kForce` 表加一行，名稱就是命令列的值：

   ```cpp
   {"dynamic", forces_dynamic},
   ```

之後就能執行 `./bh --backend omp --force dynamic --threads 8`，並用測試確認結果正確：

```bash
CONFIGS="--backend omp --force dynamic" bash tests/run_tests.sh
```

建樹方法的加法相同，表是 `kBuild`，函式簽章是 `void (const Particles&, const BBox&, Octree&)`。

### 產生器、驗證模組、計時與 CSV（莊明儒）

這幾項在提案的分工中屬於莊明儒。repo 裡現有的版本是為了讓循序版本能執行、能驗證而先寫的，可以直接沿用或改寫：

- 工作負載產生器：`src/generator.cpp`，三種分布都已實作（Plummer 球依 Aarseth 等人的方法取樣，半徑超過 20 的樣本捨棄）。
- 驗證模組：`src/validate.cpp`，內容見下面的「驗證與測試」。
- 計時與 CSV：在 `src/main.cpp`，目前只記錄各階段時間、節點數與交互作用次數，還沒有每個執行緒的工作量與 λ。

### CUDA 後端（郭庭榛）

1. 在 `src/cuda/` 寫一個繼承 `bh::Backend` 的類別，實作 `load`、五個階段與 `store`。
2. 定義 `std::unique_ptr<Backend> bh::make_cuda_backend(const Options&, const Params&)`（宣告已經在 `backend.hpp`）。
3. 以 `make CUDA=1` 編譯，執行時加 `--backend cuda`。

注意事項：

- 每個階段的函式要在返回前完成（先同步），驅動器才量得到正確的時間。
- `store()` 把位置、速度、加速度與 `cost` 複製回主機，驗證模組用的是這份資料。
- 「CPU 建樹後傳到 GPU」的混合版本需要的陣列是 `child`、`size`、`mass`、`comx`、`comy`、`comz`（各取前 `num_nodes()` 個節點）以及 `next`、`key`。CPU 的建樹與質心計算可以直接呼叫 `build_tree_seq` / `build_tree_lock` 與 `com_seq`。
- 受力計算的規則請與 `force_kernel.hpp` 一致（見下一節），驗證才比得起來。

## 演算法規則

- **開啟準則**：節點邊長為 s、節點質心到粒子 i 的距離為 d，若 s/d < θ 且該節點不包含粒子 i，就以節點的總質量與質心近似；否則展開。包含粒子 i 的內部節點一律展開。走訪到葉節點時直接計算，排除粒子 i 本身。
- **積分**：kick-drift leapfrog，`integrate(dt_kick, dt_drift)` 先做 `v += a * dt_kick` 再做 `x += v * dt_drift`。第一步的 kick 只有半步，之後速度落後位置半步；驗證模組計算能量時會補回這半步。
- **精度**：預設單精度，讓 CPU 與 GPU 的結果可以直接比較。驗證用的直接加總一律以倍精度計算。
- **以鎖插入建樹（`--build lock`）**：各執行緒由根節點往下插入自己負責的粒子，只以原子比較交換（compare-and-swap）修改子節點欄位。空欄位直接換成葉節點；遇到已有粒子的葉節點，先把欄位換成 `LOCKED`，建好分開兩個粒子所需的節點後再一次寫回；讀到 `LOCKED` 或比較交換失敗就重讀重試。節點容量不足時，整個建樹以兩倍容量重做。
- **不使用現成平行函式庫**：所有平行元件（reduction、prefix sum、radix sort、樹建構）都自行實作。

## 驗證與測試

加上 `--validate-every K` 後，每 K 步會做以下檢查：

- **精確度**：抽樣約 1,000 個粒子，各自與全部 N 個粒子以倍精度直接加總，回報加速度相對誤差的中位數、99 百分位數、最大值與整體 RMS。
- **能量漂移**：N ≤ 20000 時計算系統總能量（使用相同的軟化位能），回報相對於第一次驗證的漂移。
- **樹的結構**（加 `--check-tree`）：每個粒子恰好在一個葉節點中，而且 Morton 碼與所在位置相符；節點邊長與深度相符；只有最大深度的葉節點有多個粒子；每個節點的總質量與質心和子節點一致。

`make test` 對每一組後端與策略跑上述檢查，另外要求**結果與循序後端逐位數相同**：各策略建出同一棵樹、走訪順序也相同，所以節點數、交互作用次數、誤差與能量不應該有任何差異。θ = 0 時每個節點都會展開，結果必須等於直接加總（單精度下誤差約 1e-6）。

```bash
make test                                                   # 預設測 seq、omp、omp + lock
CONFIGS="--backend omp --force dynamic" bash tests/run_tests.sh   # 只測自己的策略
BH=./bh_double bash tests/run_tests.sh                      # 測倍精度版本
bash tests/stress_build.sh                                  # 平行建樹的壓力測試
```

循序版本在 N = 20000、單精度下的參考數值（平行版本在相同 θ 下應該相同）：

每個粒子的平均交互作用次數

| 分布 | θ = 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|
| uniform | 1389 | 446 | 201 | 85 |
| plummer | 3300 | 1259 | 603 | 292 |
| collision | 2773 | 1122 | 559 | 286 |

加速度相對誤差的 99 百分位數

| 分布 | θ = 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|
| uniform | 2.9e-3 | 1.1e-2 | 2.7e-2 | 1.2e-1 |
| plummer | 2.2e-3 | 9.7e-3 | 2.3e-2 | 6.7e-2 |
| collision | 3.1e-3 | 1.2e-2 | 2.4e-2 | 8.6e-2 |

## 目前進度

| 負責人 | 項目 | 狀態 |
|---|---|---|
| 林心綸 | 循序 Barnes-Hut（五個階段） | 完成 |
| 林心綸 | 以鎖插入建樹 | 完成 |
| 林心綸 | Morton 碼計算 | 完成（建樹時計算，存在 `tree.key`） |
| 林心綸 | 平行質心計算 | 未開始 |
| 林心綸 | CPU Morton 建樹（Karras） | 未開始，需要平行 radix sort |
| 莊明儒 | 工作負載產生器、O(N²) 直接計算、驗證模組 | 已有可用版本（見上） |
| 莊明儒 | 計時與 CSV 框架 | 已有基本版本（見上） |
| 莊明儒 | CPU 平行 prefix sum 與 radix sort | 未開始 |
| 莊明儒 | OpenMP 受力計算（static／dynamic／guided）、costzones | 未開始 |
| 郭庭榛 | CUDA 後端 | 未開始 |
| 林心綸、郭庭榛 | GPU Morton 建樹 | 未開始 |

## 初步量測

N = 10⁶、單精度，每步時間的中位數（秒）。量測環境是 Intel Core i5-14400（效能核與效率核混合，16 個邏輯處理器）上的 Docker，數字只供參考，正式結果要在實驗平台上重跑。

樹建構

| 分布 | 循序 | 以鎖插入，1 執行緒 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| uniform | 0.071 | 0.059 | 0.038 | 0.018 | 0.013 | 0.010 |
| plummer | 0.081 | 0.071 | 0.039 | 0.022 | 0.016 | 0.011 |

- 相對循序建樹，16 執行緒的加速比約 7 倍。
- 質心計算目前是循序的：在循序建出的樹上約 0.03 秒，在以鎖插入建出的樹上約 0.07 秒（節點編號沒有順序，記憶體存取較分散），已經比 4 執行緒以上的建樹慢。
- 循序版本跑 Plummer 球、θ = 0.5 的完整時間步約 25 秒，受力計算佔 99.4%。
