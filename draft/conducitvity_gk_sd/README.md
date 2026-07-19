# Ionic conductivity: Green-Kubo (velocity) vs cross-square-displacement (Einstein)

给同一条电解质 MD 轨迹，用**两种等价方法**算离子电导率并对比：

- **位移法 / Einstein（CSD）**：拟合集体电荷位移 MSD 的扩散区斜率。**稳健**。
- **速度法 / Green-Kubo（GK）**：对集体速度自相关（VACF）积分。**依赖速度存储步长**。

> **一句话结论**：对 100 fs 存储间隔的轨迹，**必须信 Einstein/CSD 的值**。GK 会严重高估（本例 ~27×），
> 原因不是 bug（代码已严格验证），而是 100 fs 采样看不到 VACF 在亚 100 fs 内的 ballistic 负谷，
> 梯形积分把 τ≈0 正峰算多了。GK 要收敛需速度每 ~1–5 fs 存一次。

---

## 本例结果（naotf_dme, 1 M, 298 K, 20 ns, 100 fs, PAINN FP32 NVT）

| 方法 | 电导率 (mS/cm) | 状态 |
|---|---|---|
| 位移 / Einstein（拟合 0.3–2.4 ns）| **5.30** | ✅ 可靠 |
| 速度 / Green-Kubo（VACF 积分, 100 fs）| **~144** | ⚠️ 100 fs 下高估，勿用 |

---

## 文件

```
compare_gk_vs_csd.py     完整版：GK（本地积分）+ CSD（调 mdcraft Onsager）+ 作图
minimal_conductivity.py  精简自包含版：只用 ase/numpy/scipy，同时算两种方法
results/
  conductivity_gk_vs_csd_naotf_dme.png       主图：VACF(τ) + running κ(τ)，GK vs CSD
  vacf_mine_vs_mdcraft.png                    验证图：我的 VACF 与 mdcraft VACF 完全重合
  conductivity_mdcraft_*_PAINN.png            mdcraft 自带的 CSD/VACF 诊断图
  gk_data_naotf_dme.npz                       原始 τ, κ(τ), acf_pp/pm/mm
minimal_conductivity_<traj>.png              minimal 脚本输出的图（存到 cwd）
```

---

## 环境

用 conda 环境 **`fairchemV2_new`**（含 `mdcraft`, `MDAnalysis`, `ase`, `scipy`, `matplotlib`）：

```bash
# NERSC 上：
source activate /pscratch/sd/y/yuejian/envs/fairchemV2_new     # 或对应路径
python -c "import mdcraft, MDAnalysis, ase, scipy; print('ok')"
```

- `minimal_conductivity.py` **只需** `ase numpy scipy matplotlib`（不需要 mdcraft，完全可移植）。
- `compare_gk_vs_csd.py` **额外需要** `mdcraft`（跑 CSD），且见下方「可移植性」。

---

## 怎么跑

### A. 精简自包含版（推荐先用这个，可移植）
```bash
python minimal_conductivity.py <轨迹.traj 或 含 traj 的目录>
```
输出：两种方法的电导率 + 一张 VACF/running-κ 图。参数在**脚本顶部常量**里改（见下）。

### B. 完整版（多一个 mdcraft CSD 交叉验证 + 更多诊断图）
```bash
python compare_gk_vs_csd.py <轨迹或目录> --out_dir ./results
# 常用可选项：
#   --cat Na --anion OTf --solvent DME   # 不传则从路径自动猜
#   --dt_fs 100 --T 298                  # 不传则从路径名(…100fs…298K…)解析
#   --gk_window_ns 3.0                   # GK 用的原生分辨率窗口（主要耗时项）
#   --no_csd / --no_gk                   # 只跑其中一种
```

---

## 换体系要改什么

**minimal_conductivity.py** 顶部常量：
```python
CATION = "Na"   # 单原子阳离子的元素符号
ANION  = "S"    # 阴离子「中心重原子」元素（每个阴离子唯一一个）。OTf→S, PF6→P, TFSI→N
DT_FS  = 100.0  # 轨迹存储步长 = GK 的积分步长（fs）——务必对
T_K    = 298.0  # 温度
```
（假设：单原子阳离子、阴离子中心原子元素唯一、正交盒、NVT 定容、1:1 电解质 z=±1。）

**compare_gk_vs_csd.py** 用命令行参数，阴离子中心原子查表 `compute.anion_central_dict`。

> ⚠️ **最影响结果的 3 个量是 species、`DT_FS`、`T`**。脚本会把它们打印在开头，跑之前**先核对**。

---

## 可移植性（复制到别的 server 后）

- **`minimal_conductivity.py`**：无 repo 依赖，随便放，只要装了 ase/scipy 即可。
- **`compare_gk_vs_csd.py`**：会 `import` repo 里的
  `APPLICATIONS/electrolytes/observable_scripts/conductivity/compute.py`
  和 `submodule/transport-coefficients/example_calculation/lij_analysis.py`。
  它靠 `_find_repo_root()` **向上层目录搜** `APPLICATIONS/.../conductivity/compute.py` 来定位 repo。
  **所以这个 draft 文件夹必须放在一份完整的 `MLFF-distill` checkout 内部**（含 submodule），完整版才能跑；
  否则会报「Cannot locate MLFF-distill repo root」——这种情况就用 minimal 版。

---

## 为什么 GK 偏高（已验证，不是 bug）

三步验证：
1. **单位/用法对**：用官方参考算例（LiCl/DMSO）复现电导率到 **13 位**（2.138097101651705，参考 2.13809710165179）。
2. **VACF 对**：本轨迹上，我的 GK VACF 与 mdcraft 的 VACF（同一份 `TransportCoefficients.compute_acf`）
   **逐点一致到机器精度**（max 差 ~8e-19，相对 ~1e-15）。见 `results/vacf_mine_vs_mdcraft.png`。
3. **偏高来自采样步长**：把 100 fs 的 VACF 再人为抽稀，GK 单调飙升 →
   100 fs: 144、200 fs: 733、300 fs: 1049、500 fs: 1769、1000 fs: 4580 mS/cm。
   反推 <100 fs 会降向 Einstein 的 5.3。

**机制**：VACF 在 τ=0 为大正值，真实曲线在 0–100 fs 内快速下探到负值再回升；100 fs 只有 τ=0 和 τ=100 fs 两个点，
梯形直线略过中间的负谷，把 τ≈0 正峰积分**算多了**。电导率是「大自相关−大交叉项」抵消后的小残差，故此偏差被放大到 ~27×。

**GK 不涉及位置/unwrap**（只用速度），所以高估与 unwrap 无关；Einstein 用的 unwrap 位置已由「= mdcraft 5.29」验证正确。

### GK 要多细的存储步长才够？
- **100 fs**：不够，严重高估（本例 27×），别用。
- **10 fs**：通常够（比 100 fs 好一到两个量级），但落在 VACF 特征时间（离子平动 ~几十–100 fs）同量级，可能残留 ~10–50%，不保证完全收敛。
- **2–5 fs**：最稳（梯形误差 ∝ dt²，快速消失）。
- **验收判据**（不管存多细都要做）：running κ(τ) 出现清晰平台，且平台值 ≈ Einstein/CSD。两者吻合=够；不吻合=还欠采样，降步长重存。
- 更保险：阴离子用**离子质心速度**（整分子质量加权）代替中心原子速度，去掉高频振动，VACF 更干净。

---

## 关键数（可用于回归测试）

- Einstein/CSD（本体系）≈ **5.29–5.30 mS/cm**
- 参考算例（LiCl/DMSO, example_output）在 20 ps 截断 = **2.138097101651… mS/cm**
- 我的 VACF vs mdcraft VACF：相对差 ~1e-15
