# Song098 Blind Video Pilot

更新日期：2026-08-20
状态：pilot ready，paper incomplete

## 1. 目的

验证同一套 renderer 能否从 generator PKL 和 SONIC recorded execution 生成无标签、同相机、
同时间窗、同音频的独立视频，并通过盲评素材审计。该 pilot 不用于论文统计，只验证素材
生产和随机化链路。

## 2. 固定配置

| 字段 | 配置 |
|---|---|
| Song | FineDance 098 sliced audio |
| Generation seed | 1234 |
| Route | M0、M2、M4 |
| Representation | reference、SONIC execution r01 |
| Source window | `[4,20)`，16 s |
| Video | 640 x 480、30 FPS、H.264 |
| Audio | identical song098 segment、AAC 192 kbps |
| Camera | MuJoCo free camera，azimuth 180、elevation -12、distance 3，pelvis-follow |
| Labels | none |

Reference 使用 PKL 中的 `root_pos/root_rot/dof_pos`。Execution 从标准 aligned run 的
`sim_state + SONIC measured body_q` 重建。两者先按同一绝对动作时间窗重采样，再渲染，
没有按模型单独调整音频或相机。

## 3. 结果

- 生成 6 个独立视频，全部为 16.000 s、640 x 480、30 FPS；
- 每段均包含一个 video stream 和一个 audio stream；
- 人工检查同一时刻 contact sheet：机器人完整可见、无 route/reference/execution 标签；
- 素材审计 6/6 eligible；
- 构造 9 个 canonical trials：generator quality 3、music match 3、execution retention 3；
- `pilot_ready=true`，`paper_ready=false`。

论文门槛仍缺：

1. 至少 3 首完整 M0/M2/M4 matched songs，目前 1 首；
2. 每首至少 3 个 generation seeds，目前 1 个；
3. 每条 reference 至少 3 个 SONIC repeats，目前 1 次。

## 4. 产物

- 原始素材：[`media/song098_seed1234_pilot`](../../eval/human_study/media/song098_seed1234_pilot)
- 素材 manifest：[`assets_song098_seed1234_pilot.json`](../../eval/human_study/assets_song098_seed1234_pilot.json)
- 公开盲化设计：[`public_study.json`](../../eval/human_study/design_song098_seed1234_pilot/public_study.json)
- 私有解盲映射：[`private_key.json`](../../eval/human_study/design_song098_seed1234_pilot/private_key.json)
- 画面检查：[`contact_sheet_t12.png`](../../eval/human_study/design_song098_seed1234_pilot/contact_sheet_t12.png)

## 5. 复现

MuJoCo 离屏渲染使用 SONIC simulation 环境；`audiomimic` 环境当前缺少可用的 EGL/OSMesa
renderer 组合。

```bash
cd ~/AudioMimic

MUJOCO_GL=egl \
  ~/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/render_song098_blind_pilot.py

python eval/build_human_pairwise_study.py \
  --assets eval/human_study/assets_song098_seed1234_pilot.json \
  --output_dir eval/human_study/design_song098_seed1234_pilot
```

公开 `media/clip_*.mp4` 使用硬链接，不复制视频数据。正式问卷不得公开 `private_key.json`。
