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
| OpenMP：樹建構，Morton 碼排序後建樹（`--build morton`） | 完成 |
| CPU 共用平行 prefix sum、穩定 radix sort | 完成 |
| OpenMP：受力計算 static / dynamic / guided / costzones | 完成 |
| Q2 負載量測、自動化實驗與繪圖 | 完成工具與本機驗證；正式硬體實驗待執行 |
| CUDA 後端 | 未開始（Makefile 的 `CUDA=1` 路徑尚未測試過） |
| Q1 實驗腳本（建樹方法比較） | 完成，本機試跑過；正式實驗待執行 |
| Q4 實驗腳本（近似參數 θ） | 完成，本機試跑過精確度與計算量；平行效率待正式實驗 |

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
./bh --backend omp --force costzones --threads 8 --n 100000 --steps 10 --warmup 2 --csv q2.csv
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
| `--force` | `omp` 後端的受力計算策略：`seq`、`static`、`dynamic`、`guided`、`costzones` | `seq` |
| `--force-order` | 受力資料排列：`morton`（排序並實際重排受力 SoA）、`original`；costzones 必須使用 Morton | `morton` |
| `--chunk` | 0 表示 static 連續平均分配、dynamic / guided 使用 chunk 1；正整數指定 chunk | 0 |
| `--threads` | `omp` 後端的執行緒數；CSV 額外記錄受力階段實際 team size | OpenMP 預設 |
| `--validate-every K` | 每 K 步與 O(N²) 直接加總比較（不計時） | 0（關閉） |
| `--check-tree` | 驗證時一併檢查樹結構 | 關閉 |
| `--max-err`、`--max-drift` | p99 相對誤差或能量漂移超過門檻時以狀態碼 2 結束 | 不檢查 |
| `--csv FILE` | 每個時間步附加一列（設定、各階段秒數、節點數、交互作用次數、驗證結果） | 不輸出 |

單位：G = 1、總質量 = 1、Plummer 尺度半徑 = 1。

Q2 的每執行緒工作量、負載不均 λ、排序／分割／走訪時間與暖機標記也會寫入 CSV。
新欄位和舊版 CSV 不相容時會拒絕附加，請使用新檔案。完整方法、共用 API、實驗與圖表命令見 [docs/q2.md](docs/q2.md)。

## 程式架構

```
include/bh/
  types.hpp         real、Params、Particles（SoA）、BBox
  octree.hpp        八元樹（SoA、無指標）與編碼方式
  backend.hpp       Backend 介面：五個階段 + load/store
  stages.hpp        CPU 各階段函式宣告
  force_kernel.hpp  單一粒子的樹走訪（所有 CPU 受力計算策略共用）
  forces.hpp        平行受力策略、costzones 分割與量測介面
  parallel.hpp      共用 CPU prefix sum、穩定 Morton radix sort
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
  test_parallel.cpp 平行元件、分割覆蓋、多步受力逐位元一致與工作量測試
  test_cli.py       CSV、CLI、實際 team size、分析公式整合測試
bench/
  com_levels.cpp    逐層質心計算的每層計時（make bench）
  radix_sort.cpp    Morton radix sort 單獨計時並核對結果（make bench-radix，見 docs/radix-sort.md）
scripts/
  run_q1.py         Q1 實驗：各建樹方法在不同分布、N、執行緒數下的建樹時間
  analyze_q1.py     彙整 Q1 結果成表格與圖
  run_q2.py         重複實驗、暖機排除、原始 CSV 與可重現環境紀錄
  analyze_q2.py     彙整 strong / weak scaling、λ、區域性與對撞時間序列圖
  run_com.py        質心計算實驗：對各分布、N、執行緒數執行 com_levels
  analyze_com.py    彙整質心計算實驗成表格與圖
  run_q4.py         Q4 實驗：各 θ 的誤差、交互作用次數與各執行緒數下的時間
  analyze_q4.py     彙整 Q4 結果成表格與圖
  experiment.py     run_*.py 共用（輸出目錄、環境變數、meta.json）
experiments/
  q1-local.json     Q1 本機設定
  q1-full.json      Q1 實驗機器設定
  q2-local.json     本機 Q2 實驗設定
  q2-full.json      正式硬體完整實驗設定（待執行）
  com-local.json    質心計算實驗設定
  q4-local.json     Q4 本機設定（只用循序受力計算）
  q4-full.json      Q4 實驗機器設定（需要平行受力計算）
```

每個時間步依序呼叫後端的五個階段，驅動器分別計時：

1. `compute_bbox()` 邊界框
2. `build_tree()` 樹建構
3. `compute_com()` 質心計算
4. `compute_forces()` 受力計算（同時記錄每個粒子的交互作用次數 `cost`）
5. `integrate(dt_kick, dt_drift)` leapfrog 積分

驗證在受力計算與積分之間進行，不列入計時。

### 新增 CPU 策略（建樹、質心計算或受力計算）

1. 建樹函式在 `include/bh/stages.hpp` 宣告；受力策略在 `include/bh/forces.hpp` 宣告：
   `ForceMetrics strategy(Particles&, const Octree&, const Params&, const ForceConfig&);`。
   受力策略須回傳實際各執行緒工作量，並將必要的排列與分割成本納入計時。
2. 在 `src/cpu/` 新增實作檔。受力計算請呼叫 `force_on_particle()`，讓各策略只差在工作分配方式。
3. 在 `src/cpu/cpu_backend.cpp` 的 `kBuild`、`kCom` 或 `kForce` 表加一行，名稱即為命令列的 `--build` / `--com` / `--force` 值。
4. 用驗證測試確認：`CONFIGS="--backend omp --force dynamic" bash tests/run_tests.sh`。
   測試會檢查誤差、能量與樹結構，以及一般分布下與循序後端的輸出一致性。
   `make test-parallel` 另在同一棵樹上逐位元比較每個粒子的加速度與 cost；
   重複 Morton key 的葉鏈順序可能因既有 lock 建樹競爭不同，不以兩次獨立建樹的逐位元結果判定排程正確性。

### Q2 實驗與圖表

~~~bash
make test
make test-cli                      # 需要 Python 3；也可設定 PYTHON=python
python3 -m pip install -r requirements-analysis.txt
python3 scripts/run_q2.py --config experiments/q2-local.json --output results/q2-local
python3 scripts/analyze_q2.py --input results/q2-local --output docs/results/q2-local
~~~

實驗會保存每次執行命令、環境、硬體、編譯器、原始資料與原始碼快照。
比較時同時檢查完整受力時間與 λ；λ 較低不一定代表整體較快。

本機已完成 609 次 Q2 實驗，原始資料、環境、圖表與結果分析見 [docs/results/q2-local/analysis.md](docs/results/q2-local/analysis.md)。

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
- **Morton 建樹（`--build morton`）**：平行計算 Morton 碼後以 `radix_order_omp()` 排序（碼相同的粒子依粒子索引排列）。排序後，同一個節點底下的粒子是連續的一段，相鄰兩個碼共同的前導八分區數就是兩者最深共同節點的深度。深度 d 的節點對應「前 d 個八分區相同、至少兩個粒子」的極大連續段，並以該段的第一個粒子 a 編號：從 a 開始的節點深度是 `before(a) + 1` 到 `shared(a)`，對各位置的節點數做 prefix sum（`exclusive_scan_omp()`）就得到每個節點的索引，接著所有節點各自獨立、平行地以二分搜尋找出子節點的範圍並填入。這個做法沿用 Karras 的觀察（排序後每個節點的範圍可由共同前綴長度獨立決定），但不另外建立二元基數樹，直接產生八元樹。碼相同的粒子落在同一個最大深度的葉節點，依排序後的順序串成鏈。粒子陣列本身目前不重排，葉節點記錄的仍是原始粒子索引。
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

## Q1 實驗（樹建構）

比較 `--build lock` 與 `--build morton` 在不同分布、N 與執行緒數下的建樹時間，以循序建樹為基準。

```bash
make
python3 scripts/run_q1.py --config experiments/q1-local.json --output results/q1-local
python3 scripts/analyze_q1.py results/q1-local
```

- **`run_q1.py`**（只用 Python 標準函式庫）：對每個分布與 N 跑一次循序後端，再跑每種建樹方法與每個執行緒數的組合；每個組合重複數次，執行順序打亂。輸出目錄必須是新的，裡面有 `steps.csv`（每個時間步一列，比 `--csv` 多一欄 `repeat`）與 `meta.json`（設定、commit、編譯器、CPU）。加上 `--dry-run` 只列出要跑的組合。
- **設定檔**：`q1-local.json` 是 N = 10⁵ 與 10⁶、重複 3 次，本機約 25 分鐘；`q1-full.json` 是 N = 10⁴ 到 10⁶、重複 5 次並綁定核心，給實驗機器用。分布、N、執行緒數、建樹方法都直接改 JSON。
- **θ 設為 1000**：建樹時間與 θ 無關，而 θ 很大時受力計算只走訪包含粒子自身的節點，N = 10⁶ 時每步從約 20 秒降到不到 1 秒，整組實驗才跑得完。這組資料因此只看樹建構與質心計算兩個階段。
- **`analyze_q1.py`**：輸出 `summary.csv`、`summary.md`（每個分布與 N 一張表）與 `build-n<N>.png`（需要 matplotlib，沒有安裝時只輸出表格）。每次執行取量測步的中位數，再取各次重複的中位數；加速比以相同分布與 N 的循序建樹為基準，`scaling` 欄則是同一種方法相對於自己 1 執行緒的倍數。

`results/` 不進版本控制。

## 質心計算實驗（樹的深度與同步成本）

量測逐層質心計算（`--com level`）在每一層花的時間，把「計算」與「在 barrier 等待」分開。

```bash
make bench
python3 scripts/run_com.py --config experiments/com-local.json --output results/com-local
python3 scripts/analyze_com.py results/com-local
```

- **`bench/com_levels.cpp`**（`make bench` 編成 `build/float/com_levels`）：產生粒子、建一棵樹，在同一棵樹上重複執行 `com_level()` 與 `com_seq()`，每一層輸出一列 CSV。也可以單獨執行：`build/float/com_levels plummer 1e6 8`。
- **計時方式**：`com_level()` 的每個執行緒在做完自己那一段節點時、以及通過該層的 barrier 之後各讀一次時間。前一段是計算時間（busy），後一段是等待時間（wait）。等待時間在各執行緒間的最小值是最後到達的執行緒付出的，也就是 barrier 本身的成本；其餘是等其他執行緒做完。只有 `com_level_profile` 指向一個 `ComLevelProfile` 時才會記錄。
- **`analyze_com.py`**：輸出 `summary.csv`、`summary.md`、`levels-n<N>.png`（各層的計算與等待時間，預設看 8 執行緒，可在命令最後指定）與 `wait-share.png`（等待占整個階段的比例）。「其他」是整個階段扣掉各層計算與等待後剩下的時間，主要是依深度排序節點。
- 量測時沒有受力計算把快取洗掉，所以計算時間比實際模擬中略為樂觀；等待時間不受影響。`degenerate` 分布有大量幾乎重合的粒子，樹會長到最大深度，用來當深度的極端情況。

## Q4 實驗（近似參數 θ）

量測 θ ∈ {0.3, 0.5, 0.7, 1.0} 對精確度、計算量與平行效率的影響。

```bash
make
python3 scripts/run_q4.py --config experiments/q4-local.json --output results/q4-local
python3 scripts/analyze_q4.py results/q4-local
```

- **`run_q4.py`**（只用 Python 標準函式庫）：對每個分布與 θ 跑三種執行，`steps.csv` 以 `phase` 欄區分，執行順序打亂。加上 `--dry-run` 只列出要跑的組合。
  - `error`：每個 N 跑一步並開啟驗證，取得相對於直接加總的誤差與交互作用次數。
  - `energy`：在較小的 N（設定檔的 `energy`，N 不能超過 20000）跑較多步並定期驗證，取得能量漂移，以及粒子演化後的誤差。
  - `timing`：不開驗證，跑循序後端與各執行緒數的 OpenMP 後端，每個組合重複數次。
- **設定檔**：`force` 與 `force_args` 決定 OpenMP 後端的受力計算策略，`force_args` 會原樣接在 `bh` 的參數後面。`q4-local.json` 用循序受力計算、不掃執行緒數（`threads` 是空的），N = 10⁴ 與 10⁵，只看精確度與計算量；`q4-full.json` 是 N = 10⁴ 到 10⁶、1 到 16 執行緒，用平行受力計算的 `--force dynamic --force-order morton`。循序後端在 N = 10⁶、θ = 0.3 時每步要 1 到 2 分鐘，整組要跑幾個小時。
- **`analyze_q4.py`**：輸出 `accuracy.csv`、`timing.csv`、`summary.md`、`tradeoff.png`（誤差對每粒子交互作用次數，需要 matplotlib）與 `efficiency-n<N>.png`（有掃執行緒數時才有）。
  - 誤差取各次驗證的中位數，能量漂移取絕對值最大的一次。
  - 時間是每次執行取量測步的中位數，再取各次重複的中位數。
  - 加速比（`*_speedup`）以相同分布、N、θ 的循序後端為基準。
  - 平行效率（`*_efficiency`）是同一種策略相對於自己 1 執行緒的加速比除以執行緒數，受力計算階段與整步各算一次。平行受力計算會依 Morton 順序走訪粒子，1 執行緒就比循序後端快，所以效率不以循序後端為基準。

## 初步量測

機器是 Intel Core i5-14400（效能核與效率核混合、16 個邏輯處理器）上的 Docker，單精度。數字只供參考，正式結果要在實驗機器上重跑。

建樹階段每步時間（毫秒），N = 10⁶，來自 `experiments/q1-local.json`（重複 3 次的中位數），Morton 建樹使用 PR #8 改寫後的平行 radix sort：

| 分布 | 循序 | 方法 | 1 執行緒 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| uniform | 78.5 | lock | 73.0 | 37.4 | 20.0 | 15.2 | 11.4 |
| | | morton | 53.6 | 28.9 | 18.9 | 13.5 | 30.8 |
| plummer | 107.2 | lock | 87.4 | 47.0 | 26.7 | 17.9 | 12.1 |
| | | morton | 52.2 | 28.5 | 17.1 | 13.4 | 29.2 |
| collision | 84.9 | lock | 72.6 | 40.4 | 24.2 | 17.0 | 13.1 |
| | | morton | 52.1 | 29.0 | 16.9 | 13.6 | 26.8 |

- **時間跳動很大**：循序建樹的 collision 三次重複落在 74 到 104 毫秒；lock 的程式沒有改，1 執行緒卻比上一次量測慢約 10 毫秒。幾毫秒的差距不要解讀。
- **lock**：8 執行緒是自己 1 執行緒的 4.3 到 4.9 倍，16 執行緒仍在變快，是 16 執行緒最快的方法。
- **morton**：1 到 8 執行緒都比 lock 快，8 執行緒約 13.5 毫秒，是自己 1 執行緒的約 4 倍。16 執行緒反而變慢而且不穩定（三次重複落在 17 到 34 毫秒）。
- **N = 10⁵**：兩者都在 1 到 5 毫秒。2 到 8 執行緒大多是 Morton 較快（collision 2 執行緒兩者幾乎相同），1 執行緒的 uniform 與 collision 是 lock 較快；16 執行緒 Morton 在 plummer 與 collision 較慢（plummer 達 12.3 毫秒）。
- **排序已不是主要瓶頸**：issue #7 / PR #8 把 `radix_order_omp()` 改成 Morton 碼與索引一起搬動。另外量測 N = 10⁶ 的 Morton 建樹，排序在 1 執行緒從 54 到 59 毫秒降到約 20 毫秒，8 執行緒從 12 到 14 毫秒降到約 5 毫秒；現在建樹其餘部分（計數、scan、填節點）在 1 執行緒約 30 毫秒、8 執行緒約 8 毫秒，占比較大。
- **16 執行緒變慢的原因尚未確認**：同一支程式內重複呼叫 Morton 建樹時，16 執行緒約 11 到 17 毫秒，但在完整模擬中慢了一倍左右。

質心計算階段每步時間的中位數（秒），N = 10⁶。`level` 各欄的樹由 `--build lock` 以相同的執行緒數建出。

| 分布 | 循序（循序建的樹） | 循序（lock 建的樹） | level 1 執行緒 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| uniform | 0.028 | 0.080 | 0.033 | 0.022 | 0.014 | 0.012 | 0.010–0.032 |
| plummer | 0.028 | 0.077 | 0.040 | 0.027 | 0.018 | 0.015 | 0.013–0.027 |

逐層質心計算的分段時間（毫秒），來自 `experiments/com-local.json`，樹由 `--build morton` 建出，8 執行緒。「計算」與「等待」是每個執行緒的平均：

| N | 分布 | 層數 | 循序 | 逐層 | 計算 | 等待 | 其中 barrier | 其他 |
|---|---|---|---|---|---|---|---|---|
| 10⁴ | uniform | 13 | 0.138 | 0.037 | 0.013 | 0.013 | 0.005 | 0.011 |
| 10⁴ | plummer | 13 | 0.148 | 0.035 | 0.014 | 0.012 | 0.005 | 0.009 |
| 10⁴ | degenerate | 21 | 0.683 | 0.219 | 0.108 | 0.039 | 0.007 | 0.073 |
| 10⁶ | uniform | 15 | 36.1 | 14.1 | 9.36 | 3.42 | 0.039 | 1.31 |
| 10⁶ | plummer | 17 | 35.5 | 14.3 | 9.82 | 3.38 | 0.063 | 1.10 |
| 10⁶ | collision | 17 | 29.8 | 12.6 | 8.91 | 2.64 | 0.070 | 1.08 |
| 10⁶ | degenerate | 21 | 97.9 | 125.8 | 106.2 | 14.4 | 0.142 | 5.20 |

- **barrier 本身很便宜**：節點很少的層每層約 0.3 到 0.5 微秒，13 到 21 層合計只有幾微秒。N = 10⁶ 時不到整個階段的 1%，樹多幾層幾乎沒有影響；N = 10⁴ 時 uniform 與 plummer 的整個階段只有 0.04 毫秒，這幾微秒才占到約 14%。
- **等待主要是在等同一層的其他執行緒**：uniform、plummer、collision 在 N = 10⁶ 時等待占 21% 到 24%（4 執行緒時是 5% 到 7%）。各執行緒分到的節點數相同，推測是效能核與效率核速度不同造成的，還沒有用綁定核心驗證。
- **計算時間本身沒有等比例縮短**：N = 10⁶ 時 1 執行緒約 36 毫秒，8 執行緒時每個執行緒仍要 9 到 10 毫秒。葉節點最多的兩三層要讀取每個粒子的質量與座標，而粒子還沒有依 Morton 順序排列，屬於隨機存取。
- **深度真正的代價是多掃幾次節點陣列**：每一層都要走過整個節點陣列中屬於該層的節點。`degenerate` 在 N = 10⁶ 時有 15 層各約 25 萬個節點，逐層版本 8 執行緒仍比循序慢（125.8 對 97.9 毫秒）；循序版本沿著節點編號只掃一次。一般分布的節點集中在兩三層，1 執行緒時逐層版本比循序慢 7% 到 33%。
- 16 執行緒時執行緒數等於邏輯處理器數，只要有一個執行緒被作業系統暫停，其他執行緒都得在 barrier 等它，時間很不穩定（uniform、N = 10⁵ 的中位數是 20 毫秒，8 執行緒時是 0.35 毫秒）。

θ 對精確度與計算量的影響，來自 `experiments/q4-local.json`，循序後端，N = 10⁵。誤差是抽樣 1000 個粒子、相對於直接加總的加速度誤差：

| 分布 | θ | 每粒子交互作用次數 | 誤差中位數 | 誤差 p99 | 受力計算每步（秒） |
|---|---|---|---|---|---|
| uniform | 0.3 | 2081 | 3.8e-4 | 1.8e-3 | 1.98 |
| | 0.5 | 609 | 1.5e-3 | 6.3e-3 | 0.54 |
| | 0.7 | 263 | 3.8e-3 | 1.9e-2 | 0.22 |
| | 1.0 | 106 | 1.3e-2 | 1.3e-1 | 0.097 |
| plummer | 0.3 | 5437 | 2.3e-4 | 2.0e-3 | 5.15 |
| | 0.5 | 1682 | 9.3e-4 | 6.3e-3 | 1.16 |
| | 0.7 | 755 | 2.6e-3 | 1.8e-2 | 0.49 |
| | 1.0 | 336 | 9.7e-3 | 5.4e-2 | 0.19 |
| collision | 0.3 | 4975 | 2.8e-4 | 1.6e-3 | 4.89 |
| | 0.5 | 1590 | 1.2e-3 | 7.3e-3 | 1.10 |
| | 0.7 | 714 | 3.2e-3 | 1.6e-2 | 0.46 |
| | 1.0 | 338 | 1.0e-2 | 5.7e-2 | 0.19 |

- **θ 從 1.0 降到 0.3**：交互作用次數變成 15 到 20 倍，誤差中位數降到 1/34 到 1/42。其中 0.5 降到 0.3 這一段，交互作用次數是 3.1 到 3.4 倍，誤差約降到 1/4。
- **相同 θ 下 N 越大越準**：N 從 10⁴ 增加到 10⁵，每粒子交互作用次數增加 1.2 到 2.5 倍，誤差中位數反而降低 25% 到 53%。
- **uniform 在 θ = 1.0 的 p99 特別大**（0.13，其他分布約 0.055）：均勻立方體中心附近的加速度幾乎互相抵銷，相對誤差被放大。
- **受力計算始終是主要階段**：N = 10⁵ 循序執行時占整步的 95.0% 到 99.9%，θ 越大占比越低，但差距只有幾個百分點。

能量漂移（200 步中絕對值最大的一次，N = 20000，dt = 0.01）：

| 分布 | θ = 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|
| plummer | 2.1e-6 | 2.6e-6 | 2.4e-5 | 9.4e-5 |
| collision | 5.8e-6 | 3.8e-5 | 1.1e-4 | 7.8e-4 |
| uniform | 1.1e-2 | 1.1e-2 | 3.6e-2 | 1.8e-1 |

- plummer 與 collision 的漂移隨 θ 增加，θ = 1.0 時是 θ = 0.3 的 45 倍與 134 倍。
- uniform 是靜止的均勻立方體，約在第 140 到 160 步塌縮到中心，漂移在那時才突然變大（θ = 0.3 時前 120 步不超過 1.3e-4）。θ = 0.3 與 0.5 的最大漂移相同，可見塌縮之後的誤差主要來自時間積分，而不是樹的近似。

平行效率要在實驗機器上以 `experiments/q4-full.json` 量測。
