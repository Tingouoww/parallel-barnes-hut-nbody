# Parallel Barnes-Hut N-body

NYCU Parallel Programming (2026 Fall) 期末專題：三維重力 Barnes-Hut N 體模擬中，樹建構與力計算的平行化（OpenMP 與 CUDA）。

## 目前進度

| 部分 | 狀態 |
|---|---|
| 共用骨架（資料結構、後端介面、驅動器、Makefile） | 完成 |
| 循序 Barnes-Hut（五個階段） | 完成 |
| 工作負載產生器（uniform / plummer / collision） | 完成 |
| 驗證模組（直接加總誤差、能量漂移、樹結構檢查） | 完成 |
| OpenMP：邊界框、積分 | 完成 |
| OpenMP：樹建構（鎖插入、Morton + Karras）、質心彙整 | 未開始 |
| OpenMP：力計算排程與 costzones | 未開始 |
| CUDA 後端 | 未開始（Makefile 的 `CUDA=1` 路徑尚未測試過） |

## 編譯與執行

需要 g++（C++17、OpenMP）與 make。

```bash
make                    # 產生 ./bh（單精度）
make PRECISION=double   # 產生 ./bh_double
make test               # 跑驗證測試
```

```bash
./bh --dist plummer --n 100000 --steps 10 --theta 0.5
./bh --dist collision --n 20000 --steps 5 --validate-every 1 --check-tree
./bh --backend omp --threads 8 --n 1000000 --steps 5 --warmup 1 --csv results.csv
```

`./bh --help` 列出所有選項。常用的：

| 選項 | 說明 | 預設 |
|---|---|---|
| `--dist` | `uniform`、`plummer`、`collision`（雙 Plummer 球對撞）、`degenerate`（測試用，含重疊粒子） | `plummer` |
| `--n`、`--seed` | 粒子數、亂數種子（相同參數在任何平台產生相同粒子） | 10000、42 |
| `--steps`、`--warmup` | 時間步數；前幾步不列入統計 | 10、0 |
| `--theta`、`--dt`、`--eps` | 開啟準則 θ、時間步長、軟化長度 | 0.5、0.01、0.01 |
| `--backend` | `seq`、`omp`、`cuda` | `seq` |
| `--build`、`--force`、`--threads` | `omp` 後端的建樹策略、力計算策略、執行緒數 | `seq`、`seq` |
| `--validate-every K` | 每 K 步與 O(N²) 直接加總比較（不計時） | 0（關閉） |
| `--check-tree` | 驗證時一併檢查樹結構 | 關閉 |
| `--max-err`、`--max-drift` | p99 相對誤差或能量漂移超過門檻時以狀態碼 2 結束 | 不檢查 |
| `--csv FILE` | 每個時間步附加一列（設定、各階段秒數、節點數、交互作用次數、驗證結果） | 不輸出 |

單位：G = 1、總質量 = 1、Plummer 尺度半徑 = 1。

## 程式架構

```
include/bh/
  types.hpp         real、Params、Particles（SoA）、BBox
  octree.hpp        八元樹（SoA、無指標）與編碼方式
  backend.hpp       Backend 介面：五個階段 + load/store
  stages.hpp        CPU 各階段函式宣告
  force_kernel.hpp  單一粒子的樹走訪（所有 CPU 力計算策略共用）
  generator.hpp     工作負載產生器
  validate.hpp      驗證模組
src/
  main.cpp          驅動器：時間步迴圈、各階段計時、CSV
  generator.cpp  validate.cpp
  cpu/              循序與 OpenMP 實作、cpu_backend.cpp（策略表）
  cuda/             CUDA 後端（待實作）
tests/run_tests.sh  驗證測試
```

每個時間步依序呼叫後端的五個階段，驅動器分別計時：

1. `compute_bbox()` 邊界框
2. `build_tree()` 樹建構
3. `summarize()` 質心彙整
4. `compute_forces()` 力計算（同時記錄每個粒子的交互作用次數 `cost`）
5. `integrate(dt_kick, dt_drift)` leapfrog 積分

驗證在力計算與積分之間進行，不列入計時。

### 新增 CPU 策略（建樹或力計算）

1. 在 `include/bh/stages.hpp` 宣告函式，簽章與既有的相同，例如
   `void forces_omp_dynamic(Particles&, const Octree&, const Params&);`
2. 在 `src/cpu/` 新增實作檔。力計算請呼叫 `force_on_particle()`，讓各策略只差在工作分配方式。
3. 在 `src/cpu/cpu_backend.cpp` 的 `kBuild` 或 `kForce` 表加一行，名稱即為命令列的 `--build` / `--force` 值。
4. 用驗證測試確認：`BACKENDS=omp EXTRA="--force dynamic" bash tests/run_tests.sh`

### 新增 CUDA 後端

在 `src/cuda/` 實作 `Backend` 的子類別，並定義
`std::unique_ptr<Backend> bh::make_cuda_backend(const Options&, const Params&)`，以 `make CUDA=1` 編譯。
每個階段的函式必須在返回前完成（先同步），驅動器才能正確計時；`store()` 把位置、速度、加速度與 `cost` 複製回主機供驗證。

## 設計說明

- **八元樹**：只儲存內部節點，節點 0 是根。`child[8 * node + k]` 為 `-1`（空）、`>= 0`（內部節點索引）或 `<= -2`（葉，`leaf_head(c)` 取得粒子索引）。八分區編號 k 的 bit 0 / 1 / 2 分別代表 x / y / z 的上半部。
- **最大深度 21**：對應 63 位元 Morton 碼的解析度。深度 21 的格子不再切分，落在同一格的粒子以 `next[]` 串成鏈；其他葉節點都只有一個粒子。各種建樹方法都遵守這個規則，因此產生的樹相同、可以互相驗證。
- **開啟準則**：`s / d < θ`，s 為節點邊長，d 為粒子到節點質心的距離；只使用單極（總質量與質心）。θ ≥ 1/√3 時，包含粒子自身的格子也可能被接受，這是此準則已知的誤差來源，會反映在 θ = 1.0 的誤差分布尾端。
- **積分**：kick-drift leapfrog。第一步只做半步 kick，之後速度落後位置半步；驗證模組計算能量時會補回半步。
- **精度**：預設單精度，讓 CPU 與 GPU 的結果可以直接比較；驗證用的直接加總一律以倍精度計算。
- **不使用平行函式庫**：所有平行元件（reduction、prefix sum、radix sort、樹建構）皆自行實作。

## 驗證

`--validate-every K` 會對均勻抽樣的粒子（預設 1000 個）以倍精度直接加總計算加速度，回報相對誤差的中位數、p99、最大值，以及整體的 RMS 誤差；N ≤ 20000 時另外計算總能量與相對於第一次驗證的漂移。θ = 0 時每個節點都會展開，結果必須與直接加總一致（單精度下約 1e-6），這是檢查樹與走訪正確性最直接的方法。

循序版本在 N = 20000、單精度下的參考數值：

| 分布 | θ | 每粒子交互作用次數 | 誤差中位數 | 誤差 p99 |
|---|---|---|---|---|
| uniform | 0.5 | 446 | 2.2e-3 | 1.1e-2 |
| uniform | 1.0 | 85 | 1.7e-2 | 1.2e-1 |
| plummer | 0.5 | 1259 | 1.4e-3 | 9.7e-3 |
| plummer | 1.0 | 292 | 1.2e-2 | 6.7e-2 |
| collision | 0.5 | 1122 | 1.7e-3 | 1.2e-2 |
| collision | 1.0 | 286 | 1.3e-2 | 8.6e-2 |

平行版本在相同 θ 下的誤差應與上表相當。
