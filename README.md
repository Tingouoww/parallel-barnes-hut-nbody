# Parallel Barnes-Hut N-body

NYCU Parallel Programming (2026 Fall) 期末專題：三維重力 Barnes-Hut N 體模擬中，樹建構與受力計算的平行化（OpenMP 與 CUDA）。

## 目前進度

| 部分 | 狀態 |
|---|---|
| 共用骨架（資料結構、後端介面、驅動器、Makefile） | 完成 |
| 循序 Barnes-Hut（五個階段） | 完成 |
| 工作負載產生器（uniform / plummer / collision） | 完成 |
| 驗證模組（直接加總誤差、能量漂移、樹結構檢查） | 完成 |
| OpenMP：邊界框、積分 | 完成 |
| OpenMP：樹建構，以 CAS 鎖插入（`--build lock`） | 完成 |
| OpenMP：質心計算，逐層平行（`--com level`） | 完成 |
| OpenMP：樹建構，Morton 碼排序後建樹（`--build morton`） | 完成；排序與 prefix sum 暫時是循序版本（`src/cpu/primitives.cpp`） |
| OpenMP：受力計算排程與 costzones | 未開始 |
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
./bh --backend omp --build lock --com level --threads 8 --n 1000000 --steps 5 --warmup 1 --csv results.csv
```

`./bh --help` 列出所有選項。常用的：

| 選項 | 說明 | 預設 |
|---|---|---|
| `--dist` | `uniform`、`plummer`、`collision`（雙 Plummer 球對撞）、`degenerate`（測試用，含重疊粒子） | `plummer` |
| `--n`、`--seed` | 粒子數、亂數種子（相同參數在任何平台產生相同粒子） | 10000、42 |
| `--steps`、`--warmup` | 時間步數；前幾步不列入統計 | 10、0 |
| `--theta`、`--dt`、`--eps` | 開啟準則 θ、時間步長、軟化長度 | 0.5、0.01、0.01 |
| `--backend` | `seq`、`omp`、`cuda` | `seq` |
| `--build` | `omp` 後端的建樹策略：`seq`、`lock`、`morton` | `seq` |
| `--com` | `omp` 後端的質心計算策略：`seq`、`level` | `seq` |
| `--force` | `omp` 後端的受力計算策略：`seq` | `seq` |
| `--threads` | `omp` 後端的執行緒數 | OpenMP 預設 |
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
  force_kernel.hpp  單一粒子的樹走訪（所有 CPU 受力計算策略共用）
  primitives.hpp    prefix sum 與 radix sort 的介面（Morton 建樹與 costzones 共用）
  generator.hpp     工作負載產生器
  validate.hpp      驗證模組
src/
  main.cpp          驅動器：時間步迴圈、各階段計時、CSV
  generator.cpp  validate.cpp
  cpu/              循序與 OpenMP 實作、cpu_backend.cpp（策略表）
  cuda/             CUDA 後端（待實作）
tests/
  run_tests.sh      驗證測試（make test）
  stress_build.sh   平行建樹的壓力測試（約 4 分鐘）
```

每個時間步依序呼叫後端的五個階段，驅動器分別計時：

1. `compute_bbox()` 邊界框
2. `build_tree()` 樹建構
3. `compute_com()` 質心計算
4. `compute_forces()` 受力計算（同時記錄每個粒子的交互作用次數 `cost`）
5. `integrate(dt_kick, dt_drift)` leapfrog 積分

驗證在受力計算與積分之間進行，不列入計時。

### 新增 CPU 策略（建樹、質心計算或受力計算）

1. 在 `include/bh/stages.hpp` 宣告函式，簽章與既有的相同，例如
   `void forces_omp_dynamic(Particles&, const Octree&, const Params&);`
2. 在 `src/cpu/` 新增實作檔。受力計算請呼叫 `force_on_particle()`，讓各策略只差在工作分配方式。
3. 在 `src/cpu/cpu_backend.cpp` 的 `kBuild`、`kCom` 或 `kForce` 表加一行，名稱即為命令列的 `--build` / `--com` / `--force` 值。
4. 用驗證測試確認：`CONFIGS="--backend omp --force dynamic" bash tests/run_tests.sh`。
   測試會檢查誤差、能量與樹結構，並要求節點數、交互作用次數、誤差與能量和循序後端逐位數相同
   （各策略建出同一棵樹、走訪順序也相同，所以結果不應有任何差異）。

### 新增 CUDA 後端

在 `src/cuda/` 實作 `Backend` 的子類別，並定義
`std::unique_ptr<Backend> bh::make_cuda_backend(const Options&, const Params&)`，以 `make CUDA=1` 編譯。
每個階段的函式必須在返回前完成（先同步），驅動器才能正確計時；`store()` 把位置、速度、加速度與 `cost` 複製回主機供驗證。

## 設計說明

- **八元樹**：只儲存內部節點，節點 0 是根。`child[8 * node + k]` 為 `-1`（空）、`>= 0`（內部節點索引）或 `<= -2`（葉，`leaf_head(c)` 取得粒子索引）。八分區編號 k 的 bit 0 / 1 / 2 分別代表 x / y / z 的上半部。
- **Morton 碼**：粒子座標在根節點的立方體內量化成每維 21 位元的整數，交錯成 63 位元的 Morton 碼，存在 `tree.key[i]`（由 `Octree::morton_key()` 計算）。粒子屬於哪個子節點由 Morton 碼決定：深度 d 的節點取 `(key >> (60 - 3 * d)) & 7`。
- **最大深度 21**：對應 63 位元 Morton 碼的解析度。深度 21 的格子不再切分，落在同一格的粒子（Morton 碼相同）以 `next[]` 串成鏈；其他葉節點都只有一個粒子。各種建樹方法都遵守這個規則，因此產生的樹相同、可以互相驗證。
- **節點陣列**：各陣列的長度是容量（跨時間步保留），有效節點是 `[0, num_nodes())`。平行建樹以每個執行緒一次領一塊（64 個）節點索引的方式配置，沒用完的索引留在範圍內成為空白節點（沒有子節點、`size` 與 `mass` 為 0、從根走不到），數量記在 `blank_nodes`。走訪整個節點陣列的程式要能略過它們。
- **鎖插入建樹（`--build lock`）**：各執行緒從根往下插入自己的粒子，只以原子 compare-and-swap 修改子節點欄位。空欄位直接 CAS 成葉；遇到已有粒子的葉，先 CAS 成 `LOCKED`，私下建好分開兩個粒子所需的節點，再以一次寫入發布；讀到 `LOCKED` 或 CAS 失敗就重讀重試。節點容量不足時整個建樹以兩倍容量重做（之後的時間步沿用）。
- **Morton 建樹（`--build morton`）**：平行計算 Morton 碼後排序（`sort_by_key()`，碼相同的粒子依粒子索引排列）。排序後，同一個節點底下的粒子是連續的一段，相鄰兩個碼共同的前導八分區數就是兩者最深共同節點的深度。深度 d 的節點對應「前 d 個八分區相同、至少兩個粒子」的極大連續段，並以該段的第一個粒子 a 編號：從 a 開始的節點深度是 `before(a) + 1` 到 `shared(a)`，對各位置的節點數做 prefix sum（`exclusive_scan()`）就得到每個節點的索引，接著所有節點各自獨立、平行地以二分搜尋找出子節點的範圍並填入。這個做法沿用 Karras 的觀察（排序後每個節點的範圍可由共同前綴長度獨立決定），但不另外建立二元基數樹，直接產生八元樹。碼相同的粒子落在同一個最大深度的葉節點，依排序後的順序串成鏈。粒子陣列本身目前不重排，葉節點記錄的仍是原始粒子索引。
- **逐層質心計算（`--com level`）**：節點的深度由邊長算出（邊長 = 根節點邊長 / 2^深度），先以平行的計數排序把節點依深度分組，再從最深的一層往上處理。同一層的節點彼此獨立，以 `omp for` 平行計算；每個迴圈結尾的 barrier 把層與層隔開，沒有節點的層直接跳過。每個節點都是依子節點編號順序、以子節點已存好的數值加總，所以結果與循序版本逐位元相同，也不受建樹方法與節點編號方式影響。
- **開啟準則**：`s / d < θ`，s 為節點邊長，d 為粒子到節點質心的距離；只使用單極（總質量與質心）。包含粒子自身的內部節點一律展開，不會被當成近似。
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

## 初步量測

建樹階段每步時間的中位數（秒），N = 10⁶，單精度。機器是 Intel Core i5-14400（效能核與效率核混合、16 個邏輯處理器）上的 Docker，數字只供參考，正式結果要在實驗機器上重跑。

| 分布 | 循序 | lock 1 執行緒 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| uniform | 0.071 | 0.059 | 0.038 | 0.018 | 0.013 | 0.010 |
| plummer | 0.081 | 0.071 | 0.039 | 0.022 | 0.016 | 0.011 |

相對循序建樹的加速比在 8 執行緒約 5 倍、16 執行緒約 7 倍。

Morton 建樹（`--build morton`）在相同條件下的時間。排序目前是循序的，單獨就要 0.025 到 0.04 秒，所以整體幾乎不隨執行緒數下降：

| 分布 | morton 1 執行緒 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| uniform | 0.057 | 0.041 | 0.035 | 0.033 | 0.037 |
| plummer | 0.055 | 0.046 | 0.045 | 0.033 | 0.048 |

排序以外的部分（計算 Morton 碼、計數與 prefix sum、填入節點）1 執行緒約 0.025 秒，8 執行緒約 0.007 秒，已經比同樣 8 執行緒的 lock 建樹（0.013 到 0.016 秒）短。換上平行 radix sort 之後才能做完整的比較。

質心計算階段每步時間的中位數（秒），條件同上。`level` 各欄的樹由 `--build lock` 以相同的執行緒數建出。

| 分布 | 循序（循序建的樹） | 循序（lock 建的樹） | level 1 執行緒 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| uniform | 0.028 | 0.080 | 0.033 | 0.022 | 0.014 | 0.012 | 0.010–0.032 |
| plummer | 0.028 | 0.077 | 0.040 | 0.027 | 0.018 | 0.015 | 0.013–0.027 |

逐層版本在 4 執行緒之後幾乎不再變快。分段計時的結果：依深度排序節點合計不到 1 毫秒，節點很少的層（含 barrier）也都在 0.1 毫秒以下，時間集中在葉節點最多的兩三層（N = 10⁶ 時 uniform 的樹有 15 層、plummer 有 18 層）。這幾層要讀取每個粒子的質量與座標，而粒子還沒有依 Morton 順序排列，屬於隨機存取，推測是受記憶體頻寬限制。16 執行緒時執行緒數等於邏輯處理器數，只要有一個執行緒被作業系統暫停，其他執行緒都得在 barrier 等它，時間因此不穩定（表中列出多次執行的範圍）。
