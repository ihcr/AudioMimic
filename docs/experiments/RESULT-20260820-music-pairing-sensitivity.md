# 现有 M0/M2/M4 的音乐配对敏感性

更新日期：2026-08-20
状态：固定轨迹诊断完成，因果消融待 checkpoint

## 1. 问题

现有 M2/M4 PKL 是已经生成完毕的动作。将这些固定动作重新与 shifted、wrong-song 或
silence 音频配对，只能回答“动作在时间上是否更适合原音乐”，不能回答“generator
是否在生成时使用了音乐”。后者必须保持采样噪声不变，并在不同音乐条件下重新生成。

本实验先完成当前数据允许的配对敏感性诊断，为后续 checkpoint 消融提供基线。

## 2. 协议

- 正确配对：song098、原始 60 s 音频时钟；
- 时间 null：每 2 s 对 song098 做循环平移，覆盖整段；
- 错误歌曲：song065 的 60 s 音频；
- 对象：song098 的 M0/M2/M4 reference，以及各 route 的三次 SONIC execution；
- 指标：固定时钟 impact/onset correlation、双向 BAS；
- rank：正确配对高于时间 null 的比例，仅作诊断，不解释为独立样本 p-value。

M0 是 unconditional 负控制。如果 M2/M4 没有明显高于 M0，不能据此声称音乐条件有效。

## 3. 结果

| 对象 | Route | N | Impact paired/null | Impact rank | BAS paired/null | BAS rank | BAS paired-wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| Reference | M0 | 2 | 0.012/0.001 | 70.7% | 0.257/0.253 | 46.6% | +0.013 |
| Reference | M2 | 3 | -0.007/0.003 | 26.4% | 0.247/0.265 | 25.3% | -0.005 |
| Reference | M4 | 3 | -0.016/0.003 | 17.2% | 0.242/0.253 | 33.3% | +0.001 |
| Execution | M0 | 3 | -0.001/0.005 | 40.2% | 0.198/0.237 | 3.4% | -0.012 |
| Execution | M2 | 3 | 0.006/0.007 | 41.4% | 0.227/0.245 | 24.1% | +0.001 |
| Execution | M4 | 3 | 0.006/0.006 | 48.3% | 0.176/0.219 | 9.2% | -0.044 |

M2/M4 reference 的正确音频在 impact 和 BAS 上都没有超过大多数时间平移，且与错误
歌曲相比没有稳定优势。SONIC execution 同样没有显示可保留的配对优势。M0 偶然获得
更高的 reference impact rank，进一步说明单条歌曲上的相关或 BAS 可以由动作周期和音乐
周期偶然产生。

因此，当前数据不支持“现有 M2/M4 动作在细粒度节奏上明显匹配 song098”的结论。该结果
不能反推音乐条件一定无效，因为没有执行生成时的条件干预，而且现有 onset/kinematic-beat
检测器可能没有捕捉模型使用的高层音乐语义。

## 4. 下一步因果消融

获得 M2 checkpoint 和在线 feature contract 后，对每个 song/seed 固定 diffusion noise，
重新生成以下四组：

1. `paired`：正确因果音乐条件；
2. `shifted`：音乐特征整体平移，运动 seed 不变；
3. `shuffled`：换成另一首歌曲的特征；
4. `silence`：音乐条件置零，其他输入不变。

至少使用 3 首 held-out songs x 3 generation seeds。只有 paired 相对三类 intervention 在
M-ONSET、M-TEMPO、M-PHASE、R@K/MMDist 或盲评上稳定提升，且 G/X 指标不退化，才可以
支持 generator 使用音乐的结论。

## 5. 复现

```bash
cd ~/AudioMimic
conda activate audiomimic
python eval/analyze_music_pairing_sensitivity.py
```

机器可读结果：
[`pairing_sensitivity.json`](../../eval/music_pairing_sensitivity/20260820/pairing_sensitivity.json)。
